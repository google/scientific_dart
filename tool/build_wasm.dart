// Copyright 2026 Google LLC
// Use of this source code is governed by a Apache-style
// license that can be found in the LICENSE file.

/// Unified Wasm32-WASI native build and test runner for the Scientific Dart
/// workspace (`ndarray`, `openblas`, and `pocketfft`).
///
/// Builds a single shared `native_math.wasm` module exporting all `@ffi.Native`
/// symbols across the workspace plus standard C allocator functions (`malloc`,
/// `calloc`, `realloc`, `free`), generates the `.dart_tool/wasm_build/run_wasm.mjs`
/// Node.js WASI loader glue, and compiles/executes Dart tests via `dart compile wasm`.
///
/// This is an in-repo development/CI tool, not a consumer-facing build: nothing
/// it produces is published and all of its outputs live under `.dart_tool/`.
///
/// ## Prerequisites
///
/// A Linux (`x86_64`/`arm64`) or macOS (`arm64`/`x86_64`) host with:
///
/// * A pinned official `wasi-sdk` release (`wasi-sdk-34`), automatically
///   downloaded and SHA-256 verified into `.dart_tool/wasm_sdk` on first run
///   (override with `--wasi-sdk-dir` or `WASI_SDK_PATH`). Provides
///   `wasm32-wasip1-clang`, `wasm32-wasip1-clang++`, `wasm-ld`, `llvm-ar`,
///   `llvm-ranlib`, `llvm-nm`, `wasi-libc`, `-fno-exceptions` `libc++`/
///   `libc++abi`, and `libclang_rt.builtins.a`.
/// * A host C compiler (`cc`/`gcc`/`clang`, or `CC` env var) for building
///   OpenBLAS's host-side `getarch` helper tools.
/// * `node` >= 20 with the `node:wasi` module.
/// * `make` and `tar` (OpenBLAS build and archive extraction).
/// * A Dart SDK whose `dart compile wasm` accepts
///   `--extra-compiler-option=--enable-experimental-ffi`.
///
/// ## Usage
///
/// Run from anywhere inside the workspace, in one of three modes:
///
/// * `dart run tool/build_wasm.dart --build-only` provisions the sysroot,
///   builds OpenBLAS and all native translation units, links
///   `native_math.wasm`, and writes `run_wasm.mjs` without compiling or
///   running any Dart code.
/// * `dart run tool/build_wasm.dart <file.dart> ...` does the above, then
///   compiles each given Dart file with `dart compile wasm` and runs it under
///   Node.js. Files ending in `_test.dart` must print a `package:test`
///   summary (`All tests passed!` or `All tests skipped.`) to count as passed.
/// * `dart run tool/build_wasm.dart --test-all` discovers and runs every
///   `*_test.dart` below `pkgs/{pocketfft,openblas,ndarray}/test` except the
///   host-only files listed in [excludedWasmTests].
///
/// `-j`/`--jobs` bounds native compilation concurrency (and `make -j`),
/// `--test-jobs` bounds the number of concurrent `dart compile wasm` + `node`
/// pipelines (each dart2wasm process uses several GB of memory), and
/// `--test-timeout` bounds the wall-clock time of each compile and each run.
///
/// Tests that need scratch files on Wasm read the compile-time define
/// `NDARRAY_TEST_TMPDIR` (`String.fromEnvironment`), which the tool points at a
/// fresh target-private directory below `.dart_tool/wasm_build/test_tmp` and
/// deletes again after each run.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:ffi' show Abi;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:crypto/crypto.dart';

/// Host-only test files (relative to workspace root) that inspect the host
/// filesystem, spawn host processes, or test `package:hooks` and therefore are
/// skipped in `--test-all` Wasm mode.
///
/// Every entry must exist on disk; [_WasmWorkspaceBuilder.discoverWasmTests]
/// fails otherwise, so the list cannot silently go stale.
const Set<String> excludedWasmTests = {
  'pkgs/pocketfft/test/build_hook_test.dart',
  'pkgs/openblas/test/build_hook_test.dart',
  'pkgs/ndarray/test/meta/build_infra_invariants_test.dart',
  'pkgs/ndarray/test/meta/codebase_invariants_test.dart',
  'pkgs/ndarray/test/meta/dtype_dispatch_invariants_test.dart',
  'pkgs/ndarray/test/meta/lifetime_invariants_test.dart',
  'pkgs/ndarray/test/meta/static_type_probes_test.dart',
};

/// Pinned official `wasi-sdk` release tag and version.
const String _wasiSdkTag = 'wasi-sdk-34';
const String _wasiSdkVersion = '34.0';

/// Official `wasi-sdk` release tarballs and SHA-256 digests per host [Abi].
const Map<Abi, ({String directoryName, String tarballName, String sha256})>
_wasiSdkTarballs = {
  Abi.linuxX64: (
    directoryName: 'wasi-sdk-$_wasiSdkVersion-x86_64-linux',
    tarballName: 'wasi-sdk-$_wasiSdkVersion-x86_64-linux.tar.gz',
    sha256: 'b761e3a0721dbae9c09a0059e5fdb2bf917d1b4a8a7b430fb3b5aafb0984b2c4',
  ),
  Abi.linuxArm64: (
    directoryName: 'wasi-sdk-$_wasiSdkVersion-arm64-linux',
    tarballName: 'wasi-sdk-$_wasiSdkVersion-arm64-linux.tar.gz',
    sha256: 'f7e243dff54d60bcc576e94d6166b69f410f2500ae4a9ceef34315be10e77971',
  ),
  Abi.macosArm64: (
    directoryName: 'wasi-sdk-$_wasiSdkVersion-arm64-macos',
    tarballName: 'wasi-sdk-$_wasiSdkVersion-arm64-macos.tar.gz',
    sha256: '9c59398106b417f8f14913380fdf0097a8cc0ff4af9eb3ce0065a859e88d49e9',
  ),
  Abi.macosX64: (
    directoryName: 'wasi-sdk-$_wasiSdkVersion-x86_64-macos',
    tarballName: 'wasi-sdk-$_wasiSdkVersion-x86_64-macos.tar.gz',
    sha256: '87d27fa8adc68dee59bfbf2e22a6d34ef717c34d6bf1d8af2a56fc929d9ce0eb',
  ),
};

/// Target triple used with `wasi-sdk`.
const String _wasiTargetTriple = 'wasm32-wasip1';

/// OpenBLAS release built for Wasm.
const String _openBlasVersion = '0.3.33';

/// Name of the top-level directory inside the release tarball, also used for
/// the staged source tree in `.dart_tool/wasm_build`.
const String _openBlasSourceDirectoryName = 'OpenBLAS-$_openBlasVersion';

/// File name of the OpenBLAS release tarball.
const String _openBlasTarballName = '$_openBlasSourceDirectoryName.tar.gz';

/// Download URL of the OpenBLAS release tarball.
const String _openBlasTarballUrl =
    'https://github.com/OpenMathLib/OpenBLAS/releases/download/'
    'v$_openBlasVersion/$_openBlasTarballName';

/// SHA-256 of the tarball at [_openBlasTarballUrl]; the same value is pinned
/// by the mainline hook in `pkgs/openblas/hook/build.dart`.
const String _openBlasTarballSha256 =
    '6761af1d9f5d353ab4f0b7497be2643313b36c8f31caec0144bfef198e71e6ab';

/// Current schema version for OpenBLAS Fortran subroutine normalization.
///
/// Bump whenever [_WasmWorkspaceBuilder._normalizeOpenBlasSubroutines]
/// changes: the staged source tree is then re-extracted from the pristine
/// tarball and OpenBLAS is rebuilt.
const String _openBlasNormalizationStampVersion =
    'v4-wasi-sdk-34-wasm128-void-subroutines-sections';

/// Current schema version for incremental `.o` compilation flags and the
/// layout of the `objs/hashes.json` cache.
const String _objectBuildSchemaVersion =
    'v4-wasi-sdk-34-wasm32-wasip1-simd128-sections-depfiles';

/// The `package:ffi` version the hand-written overlay sources in
/// [_WasmWorkspaceBuilder.ensureWasmPackageConfig] were derived from.
const String _expectedFfiPackageVersion = '2.2.0';

/// Environment variable that tells `run_wasm.mjs` to require a `package:test`
/// summary line (`All tests passed!` / `All tests skipped.`) before reporting
/// success.
const String _expectTestSummaryEnvironmentVariable =
    'NDARRAY_WASM_EXPECT_TEST_SUMMARY';

/// Compile-time define (`-D<name>=<path>`) through which Wasm tests that need
/// scratch files learn their target-private directory below
/// `.dart_tool/wasm_build/test_tmp`; see
/// [_WasmWorkspaceBuilder.runDartTargets].
const String _testTemporaryDirectoryDefine = 'NDARRAY_TEST_TMPDIR';

Future<void> main(List<String> args) async {
  final defaultTestJobs = math.max(1, Platform.numberOfProcessors ~/ 4);
  final parser = ArgParser()
    ..addFlag(
      'build-only',
      negatable: false,
      help:
          'Build native_math.wasm and run_wasm.mjs without compiling or '
          'running Dart tests.',
    )
    ..addFlag(
      'test-all',
      negatable: false,
      help:
          'Discover and run all Wasm-compatible tests across '
          'pkgs/pocketfft/test, pkgs/openblas/test, and pkgs/ndarray/test.',
    )
    ..addFlag(
      'force-rebuild',
      negatable: false,
      help:
          'Force recompilation of OpenBLAS, all native object files, and '
          'native_math.wasm, ignoring every cache stamp.',
    )
    ..addFlag(
      'allow-missing-symbols',
      negatable: false,
      help:
          'Only warn (instead of failing) when an @ffi.Native symbol has no '
          'definition in the native objects or libopenblas_wasm128.a.',
    )
    ..addOption(
      'jobs',
      abbr: 'j',
      defaultsTo: '${Platform.numberOfProcessors}',
      help:
          'Maximum number of concurrent native compile processes (also used '
          'for make -j when building OpenBLAS).',
    )
    ..addOption(
      'test-jobs',
      defaultsTo: '$defaultTestJobs',
      help:
          'Maximum number of concurrent `dart compile wasm` + node pipelines. '
          'Each dart2wasm process uses several GB of memory.',
    )
    ..addOption(
      'test-timeout',
      defaultsTo: '600',
      help:
          'Per-target timeout in seconds, applied separately to '
          '`dart compile wasm` and to the node run.',
    )
    ..addOption(
      'wasi-sdk-dir',
      help:
          'Path to an existing wasi-sdk installation (defaults to '
          'WASI_SDK_PATH or an auto-downloaded pinned $_wasiSdkTag in '
          '.dart_tool/wasm_sdk).',
    )
    ..addFlag(
      'help',
      abbr: 'h',
      negatable: false,
      help: 'Print usage information.',
    );

  final ArgResults parsed;
  try {
    parsed = parser.parse(args);
  } on FormatException catch (e) {
    stderr.writeln('Error: ${e.message}');
    _printUsage(parser, stderr);
    exitCode = 2;
    return;
  }

  if (parsed.flag('help')) {
    _printUsage(parser, stdout);
    return;
  }

  int? requirePositiveInt(String optionName) {
    final value = int.tryParse(parsed.option(optionName) ?? '');
    if (value == null || value < 1) {
      stderr.writeln('Error: --$optionName must be a positive integer.');
      _printUsage(parser, stderr);
      exitCode = 2;
      return null;
    }
    return value;
  }

  final jobs = requirePositiveInt('jobs');
  if (jobs == null) return;
  final testJobs = requirePositiveInt('test-jobs');
  if (testJobs == null) return;
  final testTimeoutSeconds = requirePositiveInt('test-timeout');
  if (testTimeoutSeconds == null) return;

  final buildOnly = parsed.flag('build-only');
  final testAll = parsed.flag('test-all');
  final forceRebuild = parsed.flag('force-rebuild');
  final allowMissingSymbols = parsed.flag('allow-missing-symbols');
  final customWasiSdkPath =
      parsed.option('wasi-sdk-dir') ?? Platform.environment['WASI_SDK_PATH'];
  final targetFiles = parsed.rest;

  if (!buildOnly && !testAll && targetFiles.isEmpty) {
    stderr.writeln(
      'Error: Specify --build-only, --test-all, or one or more Dart files to '
      'compile and run.',
    );
    _printUsage(parser, stderr);
    exitCode = 2;
    return;
  }

  try {
    final repoRoot = _findRepoRoot();
    final builder = _WasmWorkspaceBuilder(
      repoRoot: repoRoot,
      jobs: jobs,
      testJobs: testJobs,
      testTimeout: Duration(seconds: testTimeoutSeconds),
      forceRebuild: forceRebuild,
      allowMissingSymbols: allowMissingSymbols,
      customWasiSdkPath: customWasiSdkPath,
    );

    // Validate explicit targets before starting the (slow) native build.
    final explicitTargets = <String>[];
    for (final arg in targetFiles) {
      final normalized = _normalizeRepoRelativePath(repoRoot, arg);
      final target = normalized.startsWith('/')
          ? File(normalized)
          : File('${repoRoot.path}/$normalized');
      if (!target.existsSync()) {
        stderr.writeln('Error: Target Dart file not found: $arg');
        exitCode = 2;
        return;
      }
      if (!explicitTargets.contains(normalized)) {
        explicitTargets.add(normalized);
      }
    }

    await builder.buildAll();

    if (buildOnly) {
      stdout.writeln(
        'Wasm native build completed: '
        '${builder.nativeWasmFile.path} and ${builder.runWasmMjsFile.path}',
      );
      return;
    }

    final filesToRun = <String>[if (testAll) ...builder.discoverWasmTests()];
    for (final target in explicitTargets) {
      if (!filesToRun.contains(target)) {
        filesToRun.add(target);
      }
    }

    final ok = await builder.runDartTargets(filesToRun);
    if (!ok) {
      exitCode = 1;
    }
  } on _BuildException catch (e) {
    stderr.writeln('Build failed: ${e.message}');
    exitCode = 1;
  }
}

void _printUsage(ArgParser parser, IOSink sink) {
  sink.writeln(
    'Usage: dart run tool/build_wasm.dart [options] [<file.dart> ...]',
  );
  sink.writeln();
  sink.writeln(parser.usage);
}

/// Finds the workspace root (the directory containing `pubspec.yaml` and
/// `pkgs/ndarray`) by walking up from the current directory.
///
/// Throws a [_BuildException] if no ancestor directory qualifies.
Directory _findRepoRoot() {
  final start = Directory.current.absolute;
  var dir = start;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/pkgs/ndarray').existsSync()) {
      return dir;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw _BuildException(
        'Could not find the workspace root (a directory containing '
        'pubspec.yaml and pkgs/ndarray) in ${start.path} or any of its '
        'parent directories.',
      );
    }
    dir = parent;
  }
}

/// Returns [inputPath] relative to [repoRoot] when it lies inside the
/// workspace, or its normalized absolute path otherwise.
String _normalizeRepoRelativePath(Directory repoRoot, String inputPath) {
  final absolutePath = File(
    inputPath,
  ).absolute.uri.normalizePath().toFilePath();
  final rootPrefix = '${repoRoot.path}/';
  if (absolutePath.startsWith(rootPrefix)) {
    return absolutePath.substring(rootPrefix.length);
  }
  return absolutePath;
}

/// Returns the last path segment of [path].
String _baseName(String path) => path.substring(path.lastIndexOf('/') + 1);

/// Orchestrates pinned `wasi-sdk` provisioning, OpenBLAS Wasm128 compilation,
/// incremental C/C++ object compilation, `wasm-ld` linking, and Node.js Wasm
/// execution.
final class _WasmWorkspaceBuilder {
  /// Creates a builder for [repoRoot] with maximum native compile concurrency
  /// [jobs] and maximum Dart compile+run concurrency [testJobs].
  _WasmWorkspaceBuilder({
    required this.repoRoot,
    required this.jobs,
    required this.testJobs,
    required this.testTimeout,
    this.forceRebuild = false,
    this.allowMissingSymbols = false,
    this.customWasiSdkPath,
  });

  /// Workspace root directory.
  final Directory repoRoot;

  /// Maximum number of concurrent native compile processes; also passed to
  /// `make -j` for the OpenBLAS build.
  final int jobs;

  /// Maximum number of concurrent `dart compile wasm` + `node` pipelines.
  final int testJobs;

  /// Timeout applied separately to each `dart compile wasm` invocation and to
  /// each `node` run of a target.
  final Duration testTimeout;

  /// Whether to ignore every cache stamp and rebuild OpenBLAS, all translation
  /// units, and `native_math.wasm`.
  final bool forceRebuild;

  /// Whether `@ffi.Native` symbols without a native definition only produce a
  /// warning instead of failing the link.
  final bool allowMissingSymbols;

  /// Optional user-specified `wasi-sdk` installation directory.
  final String? customWasiSdkPath;

  /// Root directory for downloaded `wasi-sdk` releases.
  Directory get wasmSdkRootDir =>
      Directory('${repoRoot.path}/.dart_tool/wasm_sdk');

  ({String directoryName, String tarballName, String sha256})
  get _hostWasiSdkSpec {
    final spec = _wasiSdkTarballs[Abi.current()];
    if (spec == null) {
      throw _BuildException(
        'No pinned $_wasiSdkTag release is configured for host ABI '
        '${Abi.current()}. Pass --wasi-sdk-dir or set WASI_SDK_PATH.',
      );
    }
    return spec;
  }

  /// Active `wasi-sdk` root directory.
  Directory get wasiSdkDir {
    final custom = customWasiSdkPath?.trim();
    if (custom != null && custom.isNotEmpty) {
      return Directory(custom);
    }
    return Directory(
      '${wasmSdkRootDir.path}/${_hostWasiSdkSpec.directoryName}',
    );
  }

  /// The `share/wasi-sysroot` directory inside [wasiSdkDir].
  Directory get wasiSysrootDir =>
      Directory('${wasiSdkDir.path}/share/wasi-sysroot');

  /// Directory containing intermediate and final Wasm build artifacts.
  Directory get buildDir => Directory('${repoRoot.path}/.dart_tool/wasm_build');

  /// Directory holding stub headers (`sys/ipc.h`, `sys/shm.h`) required by
  /// OpenBLAS's `common.h` without mutating the `wasi-sdk` sysroot.
  Directory get stubIncludeDir => Directory('${buildDir.path}/stub_include');

  /// Directory containing incrementally compiled `.o` files, their depfiles,
  /// and the `hashes.json` cache.
  Directory get objectsDir => Directory('${buildDir.path}/objs');

  /// Linked standalone Wasm native module.
  File get nativeWasmFile => File('${buildDir.path}/native_math.wasm');

  /// JSON description of the inputs of the last successful link of
  /// [nativeWasmFile]; the module is relinked when it no longer matches.
  File get _linkStampFile => File('${nativeWasmFile.path}.stamp');

  /// Node.js WASI + `additionalImports` loader script.
  File get runWasmMjsFile => File('${buildDir.path}/run_wasm.mjs');

  /// Static archive for OpenBLAS built with `TARGET=WASM128_GENERIC`.
  File get openBlasArchiveFile =>
      File('${buildDir.path}/libopenblas_wasm128.a');

  File get _openBlasStampFile =>
      File('${buildDir.path}/libopenblas_wasm128.stamp');

  /// Staged (normalized) OpenBLAS source tree.
  Directory get openBlasSourceDir =>
      Directory('${buildDir.path}/$_openBlasSourceDirectoryName');

  /// Generated package_config.json with Wasm-compatible `package:ffi` overlay.
  File get wasmPackageConfigFile =>
      File('${buildDir.path}/wasm_package_config.json');

  String get _clangBin => '${wasiSdkDir.path}/bin/$_wasiTargetTriple-clang';
  String get _clangCppBin =>
      '${wasiSdkDir.path}/bin/$_wasiTargetTriple-clang++';
  String get _wasmLdBin => '${wasiSdkDir.path}/bin/wasm-ld';
  String get _llvmArBin => '${wasiSdkDir.path}/bin/llvm-ar';
  String get _llvmRanlibBin => '${wasiSdkDir.path}/bin/llvm-ranlib';
  String get _llvmNmBin => '${wasiSdkDir.path}/bin/llvm-nm';

  String get _hostCcBin {
    final envCc = Platform.environment['CC']?.trim();
    if (envCc != null && envCc.isNotEmpty) return envCc;
    for (final candidate in const [
      '/usr/bin/cc',
      '/usr/bin/gcc',
      '/usr/bin/clang',
    ]) {
      if (File(candidate).existsSync()) return candidate;
    }
    return 'cc';
  }

  String? _cachedToolchainIdentity;

  /// Resolved `clang` path plus the first line of `clang --version`.
  ///
  /// Mixed into the object cache digests, the OpenBLAS archive stamp, and the
  /// link stamp so that a compiler change invalidates all three caches.
  String get _toolchainIdentity =>
      _cachedToolchainIdentity ??= _computeToolchainIdentity();

  String _computeToolchainIdentity() {
    final ProcessResult result;
    try {
      result = Process.runSync(_clangBin, ['--version']);
    } on ProcessException catch (e) {
      throw _BuildException(
        '$_clangBin not found or failed to start: ${e.message}',
      );
    }
    if (result.exitCode != 0) {
      throw _BuildException(
        '$_clangBin --version failed (exit ${result.exitCode}):\n'
        '${result.stderr}',
      );
    }
    final firstLine = const LineSplitter()
        .convert('${result.stdout}')
        .firstWhere((line) => line.trim().isNotEmpty, orElse: () => '');
    return '$_clangBin: ${firstLine.trim()}';
  }

  /// Provisions the pinned `wasi-sdk`, builds OpenBLAS and package objects,
  /// links `native_math.wasm`, writes `run_wasm.mjs`, and validates the Wasm
  /// module.
  Future<void> buildAll() async {
    await _checkHostPrerequisites();

    buildDir.createSync(recursive: true);
    objectsDir.createSync(recursive: true);

    await ensureWasiSdk();
    await _checkWasiSdkPrerequisites();
    _ensureOpenBlasStubIncludes();
    await ensureOpenBlasArchive();
    final (:objectFiles, recompiledCount: _) = await compileNativeSources();
    await linkNativeMathWasm(objectFiles);
    writeRunWasmMjs();
    ensureWasmPackageConfig();
  }

  Future<String?> _probeCommand(
    String executable,
    List<String> arguments,
  ) async {
    try {
      final result = await Process.run(executable, arguments);
      return result.exitCode == 0 ? '${result.stdout}' : null;
    } on ProcessException {
      return null;
    }
  }

  /// Verifies that host-level external tools (`cc`, `make`, `tar`, `node >= 20`
  /// with `node:wasi`) are available before downloading or building anything.
  Future<void> _checkHostPrerequisites() async {
    final missing = <String>[];

    final hostTools = <String, String>{
      'host C compiler (CC)': _hostCcBin,
      'make': 'make',
      'tar': 'tar',
    };
    for (final MapEntry(key: name, value: executable) in hostTools.entries) {
      if (await _probeCommand(executable, ['--version']) == null) {
        missing.add('$name ($executable)');
      }
    }

    final nodeOutput = await _probeCommand('node', [
      '-e',
      "require('node:wasi'); console.log(process.versions.node);",
    ]);
    if (nodeOutput == null) {
      missing.add('node >= 20 with the node:wasi module (node)');
    } else {
      final nodeVersion = nodeOutput.trim();
      final major = int.tryParse(nodeVersion.split('.').first) ?? 0;
      if (major < 20) {
        missing.add('node >= 20 (found node $nodeVersion)');
      }
    }

    if (missing.isNotEmpty) {
      throw _BuildException(
        'Missing host prerequisites:\n  - ${missing.join('\n  - ')}\n'
        'See the Prerequisites section at the top of tool/build_wasm.dart.',
      );
    }
  }

  /// Verifies that the provisioned (or user-provided) `wasi-sdk` contains all
  /// required LLVM/WASI binaries and `-fno-exceptions` (`noeh`) sysroot
  /// libraries.
  Future<void> _checkWasiSdkPrerequisites() async {
    final missing = <String>[];

    final sdkTools = <String, String>{
      'clang': _clangBin,
      'clang++': _clangCppBin,
      'wasm-ld': _wasmLdBin,
      'llvm-ar': _llvmArBin,
      'llvm-ranlib': _llvmRanlibBin,
      'llvm-nm': _llvmNmBin,
    };
    for (final MapEntry(key: name, value: executable) in sdkTools.entries) {
      if (await _probeCommand(executable, ['--version']) == null) {
        missing.add('$name ($executable)');
      }
    }
    if (!_isWasiSdkComplete(wasiSdkDir)) {
      missing.add(
        'complete $_wasiTargetTriple noeh sysroot and builtins under '
        '${wasiSdkDir.path}',
      );
    }

    if (missing.isNotEmpty) {
      throw _BuildException(
        'Incomplete wasi-sdk installation at ${wasiSdkDir.path}:\n'
        '  - ${missing.join('\n  - ')}',
      );
    }
  }

  bool _isWasiSdkComplete(Directory sdkRoot) {
    final sysrootLib =
        '${sdkRoot.path}/share/wasi-sysroot/lib/$_wasiTargetTriple';
    return File('${sdkRoot.path}/bin/$_wasiTargetTriple-clang').existsSync() &&
        File('${sdkRoot.path}/bin/$_wasiTargetTriple-clang++').existsSync() &&
        File('${sdkRoot.path}/bin/wasm-ld').existsSync() &&
        File('$sysrootLib/crt1-reactor.o').existsSync() &&
        File('$sysrootLib/libc.a').existsSync() &&
        File('$sysrootLib/noeh/libc++.a').existsSync() &&
        File('$sysrootLib/noeh/libc++abi.a').existsSync() &&
        _findBuiltinsArchive(sdkRoot) != null;
  }

  /// Ensures [wasiSdkDir] contains a verified `wasi-sdk` installation.
  ///
  /// When `--wasi-sdk-dir` or `WASI_SDK_PATH` is set, uses that installation
  /// directly without downloading. Otherwise downloads and SHA-256-verifies the
  /// pinned `$_wasiSdkTag` release tarball into `.dart_tool/wasm_sdk`.
  Future<void> ensureWasiSdk() async {
    final custom = customWasiSdkPath?.trim();
    if (custom != null && custom.isNotEmpty) {
      if (!wasiSdkDir.existsSync()) {
        throw _BuildException(
          'Configured wasi-sdk directory does not exist: ${wasiSdkDir.path}',
        );
      }
      return;
    }

    if (_isWasiSdkComplete(wasiSdkDir)) {
      return;
    }

    final spec = _hostWasiSdkSpec;
    wasmSdkRootDir.createSync(recursive: true);

    final tarball = File('${wasmSdkRootDir.path}/${spec.tarballName}');
    final downloadUrl = Uri.parse(
      'https://github.com/WebAssembly/wasi-sdk/releases/download/'
      '$_wasiSdkTag/${spec.tarballName}',
    );
    await _downloadVerifiedFile(
      url: downloadUrl,
      destination: tarball,
      expectedSha256: spec.sha256,
      label: '$_wasiSdkTag (${spec.tarballName})',
    );

    final stagingDir = Directory(
      '${wasmSdkRootDir.path}/.extract_${DateTime.now().microsecondsSinceEpoch}_$pid',
    );
    stagingDir.createSync(recursive: true);
    try {
      stdout.writeln('Extracting ${tarball.path}...');
      await _runChecked('tar', ['-xzf', tarball.path, '-C', stagingDir.path]);
      final extractedDir = Directory(
        '${stagingDir.path}/${spec.directoryName}',
      );
      if (!_isWasiSdkComplete(extractedDir)) {
        throw _BuildException(
          'Extracting ${tarball.path} did not produce a complete wasi-sdk at '
          '${extractedDir.path}.',
        );
      }
      if (wasiSdkDir.existsSync()) {
        wasiSdkDir.deleteSync(recursive: true);
      }
      extractedDir.renameSync(wasiSdkDir.path);
    } finally {
      if (stagingDir.existsSync()) {
        stagingDir.deleteSync(recursive: true);
      }
      if (tarball.existsSync()) {
        tarball.deleteSync();
      }
    }
  }

  /// Writes stub `sys/ipc.h` and `sys/shm.h` headers required by OpenBLAS
  /// `common.h` into [stubIncludeDir] so the `wasi-sdk` sysroot stays pristine.
  void _ensureOpenBlasStubIncludes() {
    final sysDir = Directory('${stubIncludeDir.path}/sys')
      ..createSync(recursive: true);
    final ipcHeader = File('${sysDir.path}/ipc.h');
    if (!ipcHeader.existsSync()) {
      ipcHeader.writeAsStringSync('/* Stub sys/ipc.h for wasm32-wasip1 */\n');
    }
    final shmHeader = File('${sysDir.path}/shm.h');
    if (!shmHeader.existsSync()) {
      shmHeader.writeAsStringSync('/* Stub sys/shm.h for wasm32-wasip1 */\n');
    }
  }

  File? _findBuiltinsArchive([Directory? sdkRoot]) {
    final root = sdkRoot ?? wasiSdkDir;
    final libClangDir = Directory('${root.path}/lib/clang');
    if (!libClangDir.existsSync()) return null;
    for (final entity in libClangDir.listSync(
      recursive: true,
      followLinks: false,
    )) {
      if (entity is File &&
          _baseName(entity.path).startsWith('libclang_rt.builtins') &&
          entity.path.endsWith('.a') &&
          (entity.path.contains('wasm32-unknown-wasip1/') ||
              entity.path.contains('wasm32-unknown-wasi/'))) {
        return entity;
      }
    }
    return null;
  }

  /// Ensures `.dart_tool/wasm_build/libopenblas_wasm128.a` is built by the
  /// current toolchain with normalized `void` Fortran subroutine signatures.
  ///
  /// Returns whether the archive was (re)built.
  Future<bool> ensureOpenBlasArchive() async {
    final openBlasBuildRoot = openBlasSourceDir;
    final treeStampFile = File(
      '${openBlasBuildRoot.path}/.wasm_normalization_stamp',
    );
    final lapackeHeader = File(
      '${openBlasBuildRoot.path}/lapack-netlib/LAPACKE/include/lapacke.h',
    );
    final makefile = File('${openBlasBuildRoot.path}/Makefile');
    final stampMatches =
        treeStampFile.existsSync() &&
        treeStampFile.readAsStringSync().trim() ==
            _openBlasNormalizationStampVersion;

    final expectedStamp =
        '$_openBlasNormalizationStampVersion\n$_toolchainIdentity\n';
    if (!forceRebuild &&
        openBlasArchiveFile.existsSync() &&
        _openBlasStampFile.existsSync() &&
        _openBlasStampFile.readAsStringSync() == expectedStamp) {
      // Even when the archive is restored from cache, compiling
      // `pkgs/openblas/hook/*.c` requires the normalized LAPACKE headers in
      // `openBlasBuildRoot`.
      if (!lapackeHeader.existsSync() || !stampMatches) {
        if (openBlasBuildRoot.existsSync()) {
          openBlasBuildRoot.deleteSync(recursive: true);
        }
        await _stageOpenBlasSource(openBlasBuildRoot);
        stdout.writeln(
          'Normalizing OpenBLAS Fortran SUBROUTINE return types to void...',
        );
        _normalizeOpenBlasSubroutines(openBlasBuildRoot);
        treeStampFile.writeAsStringSync(
          '$_openBlasNormalizationStampVersion\n',
        );
      }
      stdout.writeln(
        'OpenBLAS archive ${openBlasArchiveFile.path} is up to date.',
      );
      return false;
    }
    // From here on any existing archive is stale; drop the stamp first so an
    // interrupted build can never leave a stale-but-stamped archive behind.
    if (_openBlasStampFile.existsSync()) {
      _openBlasStampFile.deleteSync();
    }

    final treeIsCurrent =
        openBlasBuildRoot.existsSync() &&
        makefile.existsSync() &&
        lapackeHeader.existsSync() &&
        stampMatches;
    if (openBlasBuildRoot.existsSync() && !treeIsCurrent) {
      // Source normalization is not reversible, so a tree normalized by a
      // different schema (or a header-only cache restore) is replaced by a
      // pristine extraction.
      stdout.writeln(
        'Discarding ${openBlasBuildRoot.path} (normalized by a different '
        'schema or incomplete); re-staging pristine sources...',
      );
      openBlasBuildRoot.deleteSync(recursive: true);
    }
    if (!openBlasBuildRoot.existsSync()) {
      await _stageOpenBlasSource(openBlasBuildRoot);
    } else if (forceRebuild) {
      await _cleanOpenBlasTree(openBlasBuildRoot);
    }

    stdout.writeln(
      'Normalizing OpenBLAS Fortran SUBROUTINE return types to void...',
    );
    _normalizeOpenBlasSubroutines(openBlasBuildRoot);
    treeStampFile.writeAsStringSync('$_openBlasNormalizationStampVersion\n');

    stdout.writeln(
      'Building OpenBLAS $_openBlasVersion (TARGET=WASM128_GENERIC) with '
      '-j$jobs...',
    );
    final ccFlag =
        '$_clangBin --target=$_wasiTargetTriple '
        '--sysroot=${wasiSysrootDir.path} '
        '-idirafter ${stubIncludeDir.path} '
        '-O2 -msimd128 -ffunction-sections -fdata-sections '
        '-D__EMSCRIPTEN__ -DNO_SYSV_IPC '
        '-D_WASI_EMULATED_MMAN -D_WASI_EMULATED_SIGNAL '
        '-D_WASI_EMULATED_PROCESS_CLOCKS '
        '-Wno-implicit-function-declaration -Wno-incompatible-pointer-types';

    final makeEnv = <String, String>{
      ...Platform.environment,
      'PATH': '${wasiSdkDir.path}/bin:${Platform.environment['PATH'] ?? ''}',
    };

    final commonMakeArgs = <String>[
      'CC=$ccFlag',
      'HOSTCC=$_hostCcBin',
      'HOST_CFLAGS=-DFORCE_WASM128_GENERIC -DGEMM_MULTITHREAD_THRESHOLD=4',
      'AR=$_llvmArBin',
      'RANLIB=$_llvmRanlibBin',
      'TARGET=WASM128_GENERIC',
      'BINARY=32',
      'NOFORTRAN=1',
      'C_LAPACK=1',
      'USE_THREAD=0',
      'USE_OPENMP=0',
      'NO_SHARED=1',
      'NO_LAPACKE=0',
      'ONLY_CBLAS=0',
    ];

    // The OpenBLAS build prints every compiler invocation; its output is only
    // shown (via the exception) when a step fails.
    await _runChecked(
      'make',
      ['-j$jobs', ...commonMakeArgs, 'libs'],
      workingDirectory: openBlasBuildRoot.path,
      environment: makeEnv,
      echoOutput: false,
    );

    await _runChecked(
      'make',
      ['-j$jobs', ...commonMakeArgs, 'netlib'],
      workingDirectory: openBlasBuildRoot.path,
      environment: makeEnv,
      echoOutput: false,
    );

    // Ensure LAPACKE objects are built and archived into libopenblas*.a.
    final lapackeDir = Directory(
      '${openBlasBuildRoot.path}/lapack-netlib/LAPACKE',
    );
    if (lapackeDir.existsSync()) {
      await _runChecked(
        'make',
        ['-C', lapackeDir.path, '-j$jobs', ...commonMakeArgs, 'lapacke'],
        workingDirectory: openBlasBuildRoot.path,
        environment: makeEnv,
        echoOutput: false,
      );
    }

    // Re-run archive updates with -j1 so no parallel sub-make llvm-ar calls
    // can race on libopenblas*.a.
    await _runChecked(
      'make',
      ['-j1', ...commonMakeArgs, 'libs', 'netlib'],
      workingDirectory: openBlasBuildRoot.path,
      environment: makeEnv,
      echoOutput: false,
    );

    File? builtArchive;
    for (final entity in openBlasBuildRoot.listSync(followLinks: false)) {
      if (entity is File &&
          _baseName(entity.path).startsWith('libopenblas') &&
          entity.path.endsWith('.a')) {
        builtArchive = entity;
        break;
      }
    }
    if (builtArchive == null) {
      throw _BuildException(
        'OpenBLAS build completed but no libopenblas*.a archive was found in '
        '${openBlasBuildRoot.path}.',
      );
    }

    builtArchive.copySync(openBlasArchiveFile.path);
    _openBlasStampFile.writeAsStringSync(expectedStamp);
    stdout.writeln('Built ${openBlasArchiveFile.path}.');
    return true;
  }

  /// Stages a pristine OpenBLAS source tree at [destination] from a tarball
  /// whose SHA-256 matches [_openBlasTarballSha256].
  ///
  /// A tarball already present in `.dart_tool/wasm_build` or in the in-repo
  /// `.dart_tool/hooks_runner/shared/openblas/build` cache is reused when its
  /// checksum verifies; a corrupt or truncated copy in the build directory is
  /// deleted and the release is downloaded again.
  Future<void> _stageOpenBlasSource(Directory destination) async {
    final tarball = File('${buildDir.path}/$_openBlasTarballName');
    if (tarball.existsSync()) {
      final actualSha256 = await _sha256OfFile(tarball);
      if (actualSha256 != _openBlasTarballSha256) {
        stdout.writeln(
          'Discarding ${tarball.path}: SHA-256 $actualSha256 does not match '
          '$_openBlasTarballSha256 (truncated or corrupt download).',
        );
        tarball.deleteSync();
      }
    }

    if (!tarball.existsSync()) {
      final cachedTarball = _findCachedOpenBlasTarball();
      if (cachedTarball != null &&
          await _sha256OfFile(cachedTarball) == _openBlasTarballSha256) {
        stdout.writeln('Reusing verified ${cachedTarball.path}...');
        cachedTarball.copySync(tarball.path);
      } else {
        await _downloadOpenBlasTarball(tarball);
      }
    }

    stdout.writeln('Extracting ${tarball.path}...');
    await _runChecked('tar', ['-xzf', tarball.path, '-C', buildDir.path]);
    if (!destination.existsSync()) {
      throw _BuildException(
        'Extracting ${tarball.path} did not produce ${destination.path}.',
      );
    }
    await _cleanOpenBlasTree(destination);
  }

  /// Returns the OpenBLAS release tarball cached by the mainline
  /// `package:hooks` build in this workspace, if any.
  File? _findCachedOpenBlasTarball() {
    final cacheDir = Directory(
      '${repoRoot.path}/.dart_tool/hooks_runner/shared/openblas/build',
    );
    if (!cacheDir.existsSync()) return null;
    for (final entity in cacheDir.listSync(
      recursive: true,
      followLinks: false,
    )) {
      if (entity is File && _baseName(entity.path) == _openBlasTarballName) {
        return entity;
      }
    }
    return null;
  }

  /// Downloads [_openBlasTarballUrl] to [tarball] and verifies its SHA-256.
  Future<void> _downloadOpenBlasTarball(File tarball) => _downloadVerifiedFile(
    url: Uri.parse(_openBlasTarballUrl),
    destination: tarball,
    expectedSha256: _openBlasTarballSha256,
    label: 'OpenBLAS $_openBlasVersion',
  );

  /// Downloads [url] (following HTTPS redirects only) to `<destination>.part`,
  /// verifies its SHA-256 matches [expectedSha256], and atomically renames it
  /// to [destination].
  Future<void> _downloadVerifiedFile({
    required Uri url,
    required File destination,
    required String expectedSha256,
    required String label,
  }) async {
    stdout.writeln('Downloading $label from $url...');
    final partFile = File('${destination.path}.part.$pid');
    final client = HttpClient()
      ..connectionTimeout = const Duration(seconds: 30);
    try {
      var currentUrl = url;
      HttpClientResponse? response;
      for (var redirect = 0; redirect < 8; redirect++) {
        if (currentUrl.scheme != 'https') {
          throw _BuildException('Refusing non-HTTPS download URL: $currentUrl');
        }
        final request = await client.getUrl(currentUrl);
        request.followRedirects = false;
        final candidate = await request.close();
        if (candidate.isRedirect) {
          final location = candidate.headers.value(HttpHeaders.locationHeader);
          await candidate.drain<void>();
          if (location == null || location.isEmpty) {
            throw _BuildException(
              'Redirect without Location header while downloading $currentUrl.',
            );
          }
          currentUrl = currentUrl.resolve(location);
          continue;
        }
        response = candidate;
        break;
      }
      if (response == null) {
        throw _BuildException('Too many redirects while downloading $url.');
      }
      if (response.statusCode != 200) {
        await response.drain<void>();
        throw _BuildException(
          'Failed to download $url (HTTP ${response.statusCode}).',
        );
      }
      await response.pipe(partFile.openWrite());
    } on IOException catch (e) {
      if (partFile.existsSync()) partFile.deleteSync();
      throw _BuildException('Failed to download $url: $e');
    } finally {
      client.close();
    }

    final actualSha256 = await _sha256OfFile(partFile);
    if (actualSha256 != expectedSha256) {
      partFile.deleteSync();
      throw _BuildException(
        'SHA-256 mismatch for $url: expected $expectedSha256, got $actualSha256.',
      );
    }
    partFile.renameSync(destination.path);
  }

  static Future<String> _sha256OfFile(File file) async =>
      (await sha256.bind(file.openRead()).first).toString();

  Future<void> _cleanOpenBlasTree(Directory openBlasDir) async {
    // A pristine tree has no Makefile.conf yet, so `make clean` may fail;
    // the explicit deletions below cover that case.
    await _run('make', ['clean'], workingDirectory: openBlasDir.path);
    for (final name in [
      'Makefile.conf',
      'Makefile.conf_last',
      'config.h',
      'config_last.h',
    ]) {
      final f = File('${openBlasDir.path}/$name');
      if (f.existsSync()) f.deleteSync();
    }
    for (final entity in openBlasDir.listSync(
      recursive: true,
      followLinks: false,
    )) {
      if (entity is File) {
        final path = entity.path;
        if (path.endsWith('.o') ||
            path.endsWith('.a') ||
            path.endsWith('.so') ||
            path.contains('.so.')) {
          entity.deleteSync();
        }
      }
    }
  }

  void _normalizeOpenBlasSubroutines(Directory openBlasDir) {
    // 1. interface/lapack/*.c
    final interfaceLapackDir = Directory(
      '${openBlasDir.path}/interface/lapack',
    );
    if (interfaceLapackDir.existsSync()) {
      for (final entity in interfaceLapackDir.listSync()) {
        if (entity is File && entity.path.endsWith('.c')) {
          final original = entity.readAsStringSync();
          final updated = original
              .replaceAll('int NAME(', 'void NAME(')
              .replaceAll('return 0;', 'return;');
          if (updated != original) {
            entity.writeAsStringSync(updated);
          }
        }
      }
    }

    // 2. common_interface.h
    final commonInterfaceFile = File('${openBlasDir.path}/common_interface.h');
    if (commonInterfaceFile.existsSync()) {
      final original = commonInterfaceFile.readAsStringSync();
      var content = original.replaceAll(
        'int    BLASFUNC(xerbla)',
        'void   BLASFUNC(xerbla)',
      );
      const lapackMarker = '/* Lapack routines */';
      final markerIndex = content.indexOf(lapackMarker);
      if (markerIndex != -1) {
        final before = content.substring(0, markerIndex);
        final after = content
            .substring(markerIndex)
            .replaceAll('int BLASFUNC(', 'void BLASFUNC(');
        content = '$before$after';
      }
      if (content != original) {
        commonInterfaceFile.writeAsStringSync(content);
      }
    }

    // 3. driver/others/xerbla.c
    final xerblaFile = File('${openBlasDir.path}/driver/others/xerbla.c');
    if (xerblaFile.existsSync()) {
      final original = xerblaFile.readAsStringSync();
      final updated = original
          .replaceAll('int __xerbla', 'void __xerbla')
          .replaceAll('int BLASFUNC(xerbla)', 'void BLASFUNC(xerbla)')
          .replaceAll('return 0;', 'return;');
      if (updated != original) {
        xerblaFile.writeAsStringSync(updated);
      }
    }

    // 4. lapack-netlib/SRC/*.c and lapack-netlib/INSTALL/{dlamch,slamch,ilaver}.c
    final lapackCFiles = <File>[];
    final lapackSourceDir = Directory('${openBlasDir.path}/lapack-netlib/SRC');
    if (lapackSourceDir.existsSync()) {
      for (final entity in lapackSourceDir.listSync()) {
        if (entity is File && entity.path.endsWith('.c')) {
          lapackCFiles.add(entity);
        }
      }
    }
    for (final installName in ['dlamch.c', 'slamch.c', 'ilaver.c']) {
      final f = File('${openBlasDir.path}/lapack-netlib/INSTALL/$installName');
      if (f.existsSync()) {
        lapackCFiles.add(f);
      }
    }

    final twoArgXerblaDeclRegex = RegExp(
      r'xerbla_\(\s*char\s*\*\s*,\s*integer\s*\*\s*\)',
    );
    final twoArgXerblaCallRegex = RegExp(
      r'xerbla_\(\s*"([^"]+)"\s*,\s*(&[a-zA-Z0-9_.\[\]+-]+)\s*\)',
    );

    for (final file in lapackCFiles) {
      final original = file.readAsStringSync();
      // Preserve any inline helper preamble before "translated by f2c".
      final f2cIndex = original.indexOf('translated by f2c');
      final splitIndex = f2cIndex != -1 ? f2cIndex : 0;
      final prefix = original.substring(0, splitIndex);
      var body = original.substring(splitIndex);

      if (body.contains('/* Subroutine */ int ')) {
        body = body
            .replaceAll('/* Subroutine */ int ', '/* Subroutine */ void ')
            .replaceAll('return 0;', 'return;');
      }
      body = body
          .replaceAll(
            'extern /* Subroutine */ int ',
            'extern /* Subroutine */ void ',
          )
          .replaceAll('extern int ', 'extern void ');

      // Normalize 2-argument xerbla_ declarations and calls to 3-argument f2c
      // signature (char *, integer *, ftnlen).
      body = body.replaceAll(
        twoArgXerblaDeclRegex,
        'xerbla_(char *, integer *, ftnlen)',
      );
      body = body.replaceAllMapped(twoArgXerblaCallRegex, (match) {
        final routineName = match.group(1)!;
        final infoArg = match.group(2)!;
        return 'xerbla_("$routineName", $infoArg, (ftnlen)${routineName.length})';
      });

      final updated = '$prefix$body';
      if (updated != original) {
        file.writeAsStringSync(updated);
      }
    }

    // 5. lapack-netlib/LAPACKE/include/lapack.h
    final lapackHeader = File(
      '${openBlasDir.path}/lapack-netlib/LAPACKE/include/lapack.h',
    );
    if (lapackHeader.existsSync()) {
      final original = lapackHeader.readAsStringSync();
      final lapackSubroutineDecl = RegExp(
        r'^lapack_int\s+(LAPACK_[a-zA-Z0-9_]+\s*\()',
        multiLine: true,
      );
      final updated = original.replaceAllMapped(
        lapackSubroutineDecl,
        (m) => 'void ${m.group(1)!}',
      );
      if (updated != original) {
        lapackHeader.writeAsStringSync(updated);
      }
    }
  }

  /// Incrementally compiles all `ndarray`, `pocketfft`, and `openblas` native
  /// C/C++ translation units to `.dart_tool/wasm_build/objs/*.o`.
  ///
  /// Every unit is compiled with `-MD -MF <object>.d`. The headers listed in
  /// the resulting depfile (including sysroot headers) are recorded in
  /// `objs/hashes.json` together with a digest of their contents, so a unit is
  /// up to date only if its object exists, its schema/toolchain/flags/source
  /// digest matches, and the recorded header set still has the same contents.
  Future<({List<File> objectFiles, int recompiledCount})>
  compileNativeSources() async {
    final cpuCheckSource = _writeGeneratedCpuCheckSource();
    final units = _collectTranslationUnits(cpuCheckSource);

    final hashesFile = File('${objectsDir.path}/hashes.json');
    final previousEntries = forceRebuild
        ? <String, _ObjectCacheEntry>{}
        : _readObjectCache(hashesFile);
    // Memoizes per-header digests across units; most units share the same
    // sysroot headers.
    final fileDigestCache = <String, String?>{};

    final updatedEntries = <String, _ObjectCacheEntry>{};
    final toCompile =
        <({_TranslationUnit unit, File objectFile, String digest})>[];
    final objectFiles = <File>[];
    final reasons = <String, String>{};

    for (final unit in units) {
      final objectFile = File('${objectsDir.path}/${unit.objectName}');
      objectFiles.add(objectFile);
      final unitDigest = _computeUnitDigest(unit);
      final previous = previousEntries[unit.objectName];

      final String? reason;
      if (forceRebuild) {
        reason = '--force-rebuild';
      } else if (!objectFile.existsSync()) {
        reason = 'object missing';
      } else if (previous == null) {
        reason = 'not in cache';
      } else if (previous.digest != unitDigest) {
        reason = 'source, flags, or toolchain changed';
      } else if (previous.headersDigest !=
          _computeHeadersDigest(previous.headers, fileDigestCache)) {
        reason = 'included header changed';
      } else {
        reason = null;
      }

      if (reason == null) {
        updatedEntries[unit.objectName] = previous!;
      } else {
        reasons[unit.objectName] = reason;
        toCompile.add((unit: unit, objectFile: objectFile, digest: unitDigest));
      }
    }

    void writeCache() {
      hashesFile.writeAsStringSync(
        const JsonEncoder.withIndent('  ').convert({
          for (final MapEntry(:key, :value) in updatedEntries.entries)
            key: value.toJson(),
        }),
      );
    }

    if (toCompile.isEmpty) {
      writeCache();
      stdout.writeln(
        'All ${objectFiles.length} native Wasm object files are up to date.',
      );
      return (objectFiles: objectFiles, recompiledCount: 0);
    }

    stdout.writeln(
      'Compiling ${toCompile.length} of ${objectFiles.length} native Wasm '
      'translation units (-j$jobs):',
    );
    for (final item in toCompile) {
      stdout.writeln(
        '  ${item.unit.objectName} (${reasons[item.unit.objectName]})',
      );
    }

    try {
      await _runPool(toCompile, jobs, (item) async {
        final headers = await _compileSingleUnit(item.unit, item.objectFile);
        updatedEntries[item.unit.objectName] = _ObjectCacheEntry(
          digest: item.digest,
          headers: headers,
          headersDigest: _computeHeadersDigest(headers, fileDigestCache),
        );
      });
    } finally {
      writeCache();
    }
    return (objectFiles: objectFiles, recompiledCount: toCompile.length);
  }

  static Map<String, _ObjectCacheEntry> _readObjectCache(File hashesFile) {
    if (!hashesFile.existsSync()) return {};
    final Object? decoded;
    try {
      decoded = jsonDecode(hashesFile.readAsStringSync());
    } on FormatException {
      // A corrupted cache simply causes a full recompile.
      return {};
    }
    if (decoded is! Map<String, Object?>) return {};
    final entries = <String, _ObjectCacheEntry>{};
    for (final MapEntry(:key, :value) in decoded.entries) {
      if (_ObjectCacheEntry.tryParse(value) case final entry?) {
        entries[key] = entry;
      }
    }
    return entries;
  }

  /// Digest over [headers] (sorted absolute paths) and their current contents.
  ///
  /// Per-file digests are memoized in [fileDigestCache] (`null` marks a
  /// missing file). A missing header yields a value that never matches a
  /// stored digest, forcing recompilation.
  static String _computeHeadersDigest(
    List<String> headers,
    Map<String, String?> fileDigestCache,
  ) {
    final bytesBuilder = BytesBuilder(copy: false);
    for (final header in headers) {
      final fileDigest = fileDigestCache.putIfAbsent(header, () {
        final file = File(header);
        return file.existsSync()
            ? sha256.convert(file.readAsBytesSync()).toString()
            : null;
      });
      if (fileDigest == null) {
        return 'missing:$header';
      }
      bytesBuilder
        ..add(utf8.encode(header))
        ..addByte(0)
        ..add(utf8.encode(fileDigest))
        ..addByte(0);
    }
    return sha256.convert(bytesBuilder.takeBytes()).toString();
  }

  File _writeGeneratedCpuCheckSource() {
    final file = File('${buildDir.path}/cpu_check.c');
    const content = '''
#include <stdint.h>

int64_t ndarray_x86_cpu_features(void) {
  return 0x3F;
}

int64_t ndarray_x86_required_features(void) {
  return 0;
}

int64_t ndarray_check_x86_64_v3(void) {
  return 1;
}

__attribute__((weak))
int __cxa_thread_atexit(void (*dtor)(void *), void *obj, void *dso_symbol) {
  (void)dtor;
  (void)obj;
  (void)dso_symbol;
  return 0;
}
''';
    if (!file.existsSync() || file.readAsStringSync() != content) {
      file.writeAsStringSync(content);
    }
    return file;
  }

  List<_TranslationUnit> _collectTranslationUnits(File cpuCheckSource) {
    final units = <_TranslationUnit>[];

    final cSysrootFlags = <String>[
      '--target=$_wasiTargetTriple',
      '--sysroot=${wasiSysrootDir.path}',
      '-idirafter',
      stubIncludeDir.path,
      '-O3',
      '-msimd128',
      '-ffunction-sections',
      '-fdata-sections',
      '-D_WASI_EMULATED_MMAN',
      '-D_WASI_EMULATED_SIGNAL',
      '-D_WASI_EMULATED_PROCESS_CLOCKS',
    ];

    final cppSysrootFlags = <String>[
      '--target=$_wasiTargetTriple',
      '--sysroot=${wasiSysrootDir.path}',
      '-O3',
      '-std=c++17',
      '-fno-exceptions',
      '-fno-rtti',
      '-msimd128',
      '-ffunction-sections',
      '-fdata-sections',
      '-D_WASI_EMULATED_MMAN',
      '-D_WASI_EMULATED_SIGNAL',
      '-D_WASI_EMULATED_PROCESS_CLOCKS',
    ];

    final ndarrayRootDir = '${repoRoot.path}/pkgs/ndarray';
    final ndarrayHookDir = '$ndarrayRootDir/hook';
    final minizDir = '$ndarrayRootDir/third_party/miniz';
    final highwayDir = '$ndarrayRootDir/third_party/highway';

    final ndarrayIncludeAndDefines = <String>[
      '-DHWY_COMPILE_ONLY_STATIC',
      '-DHWY_DISABLED_TARGETS=(HWY_WASM_EMU256)',
      '-I$ndarrayRootDir',
      '-I$ndarrayHookDir',
      '-I$minizDir',
      '-I$highwayDir',
    ];

    final ndarrayCFlags = <String>[
      ...cSysrootFlags,
      ...ndarrayIncludeAndDefines,
      '-Dfopen64=fopen',
      '-Dfseeko64=fseeko',
      '-Dftello64=ftello',
      '-Dstat64=stat',
    ];

    final ndarrayCppFlags = <String>[
      ...cppSysrootFlags,
      ...ndarrayIncludeAndDefines,
    ];

    // 1. Generated cpu_check.c
    units.add(
      _TranslationUnit(
        sourceFile: cpuCheckSource,
        objectName: 'ndarray_cpu_check.o',
        isCpp: false,
        flags: ndarrayCFlags,
      ),
    );

    // 2. miniz.c
    units.add(
      _TranslationUnit(
        sourceFile: File('$minizDir/miniz.c'),
        objectName: 'ndarray_miniz.o',
        isCpp: false,
        flags: ndarrayCFlags,
      ),
    );

    // 3. Highway core + vqsort sources
    const hwyCoreFiles = [
      'targets.cc',
      'per_target.cc',
      'aligned_allocator.cc',
      'print.cc',
      'timer.cc',
    ];
    for (final name in hwyCoreFiles) {
      final file = File('$highwayDir/hwy/$name');
      if (!file.existsSync()) continue;
      final stem = name.substring(0, name.length - 3);
      units.add(
        _TranslationUnit(
          sourceFile: file,
          objectName: 'hwy_$stem.o',
          isCpp: true,
          flags: ndarrayCppFlags,
        ),
      );
    }

    final sortDir = Directory('$highwayDir/hwy/contrib/sort');
    final sortSources = sortDir.listSync().whereType<File>().where((f) {
      final name = _baseName(f.path);
      return name == 'vqsort.cc' ||
          (name.startsWith('vqsort_') && name.endsWith('.cc'));
    }).toList()..sort((a, b) => a.path.compareTo(b.path));

    for (final file in sortSources) {
      final name = _baseName(file.path);
      final stem = name.substring(0, name.length - 3);
      units.add(
        _TranslationUnit(
          sourceFile: file,
          objectName: 'hwy_$stem.o',
          isCpp: true,
          flags: ndarrayCppFlags,
        ),
      );
    }

    // 4. pkgs/ndarray/hook/*.cpp
    final ndarraySources =
        Directory(ndarrayHookDir)
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.cpp'))
            .toList()
          ..sort((a, b) => a.path.compareTo(b.path));

    for (final file in ndarraySources) {
      final name = _baseName(file.path);
      final stem = name.substring(0, name.length - 4);
      units.add(
        _TranslationUnit(
          sourceFile: file,
          objectName: 'ndarray_$stem.o',
          isCpp: true,
          flags: ndarrayCppFlags,
        ),
      );
    }

    // 5. pkgs/pocketfft/hook/pocketfft_wrapper.cpp
    final pocketfftHookDir = '${repoRoot.path}/pkgs/pocketfft/hook';
    units.add(
      _TranslationUnit(
        sourceFile: File('$pocketfftHookDir/pocketfft_wrapper.cpp'),
        objectName: 'pocketfft_wrapper.o',
        isCpp: true,
        flags: <String>[
          ...cppSysrootFlags,
          '-DPOCKETFFT_NO_MULTITHREADING',
          '-I$pocketfftHookDir',
        ],
      ),
    );

    // 6. pkgs/openblas/hook/*.c (custom_extensions.c)
    final openblasHookDir = Directory('${repoRoot.path}/pkgs/openblas/hook');
    final openBlasSourcePath = openBlasSourceDir.path;
    if (openblasHookDir.existsSync()) {
      final openblasCFiles =
          openblasHookDir
              .listSync()
              .whereType<File>()
              .where((f) => f.path.endsWith('.c'))
              .toList()
            ..sort((a, b) => a.path.compareTo(b.path));
      for (final file in openblasCFiles) {
        final name = _baseName(file.path);
        final stem = name.substring(0, name.length - 2);
        units.add(
          _TranslationUnit(
            sourceFile: file,
            objectName: 'openblas_$stem.o',
            isCpp: false,
            flags: <String>[
              ...cSysrootFlags,
              '-DADD_',
              '-DHAVE_LAPACK_CONFIG_H',
              '-DLAPACK_COMPLEX_STRUCTURE',
              '-I${openblasHookDir.path}',
              '-I$openBlasSourcePath',
              '-I$openBlasSourcePath/lapack-netlib/LAPACKE/include',
            ],
          ),
        );
      }
    }

    return units;
  }

  /// Digest of the cache schema, the toolchain, the compile flags, and the
  /// source contents of [unit]; header contents are tracked separately via
  /// the depfile (see [compileNativeSources]).
  String _computeUnitDigest(_TranslationUnit unit) {
    final bytesBuilder = BytesBuilder(copy: false)
      ..add(utf8.encode(_objectBuildSchemaVersion))
      ..addByte(0)
      ..add(utf8.encode(_toolchainIdentity))
      ..addByte(0)
      ..add(utf8.encode(unit.flags.join(' ')))
      ..addByte(0)
      ..add(unit.sourceFile.readAsBytesSync());
    return sha256.convert(bytesBuilder.takeBytes()).toString();
  }

  /// Compiles [unit] to [objectFile] and returns the sorted absolute paths of
  /// every header the compiler read, taken from the generated depfile.
  ///
  /// Throws a [_BuildException] containing the compiler output on failure.
  Future<List<String>> _compileSingleUnit(
    _TranslationUnit unit,
    File objectFile,
  ) async {
    final compiler = unit.isCpp ? _clangCppBin : _clangBin;
    final depfile = File('${objectFile.path}.d');
    final result = await _run(compiler, [
      ...unit.flags,
      '-MD',
      '-MF',
      depfile.path,
      '-c',
      unit.sourceFile.path,
      '-o',
      objectFile.path,
    ]);
    if (result.exitCode != 0) {
      throw _BuildException(
        'Failed to compile ${unit.sourceFile.path}:\n'
        '${result.stdout}\n${result.stderr}',
      );
    }
    if (!depfile.existsSync()) {
      throw _BuildException(
        '$compiler did not write the dependency file ${depfile.path} for '
        '${unit.sourceFile.path}.',
      );
    }
    return _parseDepfileHeaders(depfile.readAsStringSync(), unit.sourceFile);
  }

  /// Parses the prerequisites of a Make-style dependency file written by
  /// `clang -MD` and returns their sorted, de-duplicated absolute paths,
  /// excluding [sourceFile] itself.
  ///
  /// Handles backslash-newline continuations and the escapes clang emits for
  /// special characters: `\ ` (space), `\#`, `\\`, and `$$`.
  static List<String> _parseDepfileHeaders(String content, File sourceFile) {
    final joined = content.replaceAll('\\\r\n', ' ').replaceAll('\\\n', ' ');
    final separatorIndex = joined.indexOf(RegExp(r':(?=\s|$)'));
    final prerequisites = separatorIndex == -1
        ? joined
        : joined.substring(separatorIndex + 1);

    final paths = <String>{};
    final current = StringBuffer();
    void flush() {
      if (current.isNotEmpty) {
        paths.add(current.toString());
        current.clear();
      }
    }

    for (var i = 0; i < prerequisites.length; i++) {
      final char = prerequisites[i];
      final next = i + 1 < prerequisites.length ? prerequisites[i + 1] : '';
      if (char == r'\' && (next == ' ' || next == '#' || next == r'\')) {
        current.write(next);
        i++;
      } else if (char == r'$' && next == r'$') {
        current.write(r'$');
        i++;
      } else if (char == ' ' || char == '\t' || char == '\n' || char == '\r') {
        flush();
      } else {
        current.write(char);
      }
    }
    flush();

    final sourcePaths = {sourceFile.path, sourceFile.absolute.path};
    final workingDirectory = Directory.current.path;
    final headers = <String>{
      for (final path in paths)
        if (!sourcePaths.contains(path))
          path.startsWith('/') ? path : '$workingDirectory/$path',
    }.toList()..sort();
    return headers;
  }

  /// Links all object files, `libopenblas_wasm128.a`, and `wasi-libc` into
  /// `.dart_tool/wasm_build/native_math.wasm`, unless the link stamp shows
  /// that every link input (object digests, archive digest, export list, link
  /// arguments, `wasm-ld` path, toolchain) is unchanged since the last
  /// successful link.
  ///
  /// After a relink the module is validated with [validateNativeMathWasm]
  /// before the stamp is written. Returns whether the module was relinked.
  Future<bool> linkNativeMathWasm(List<File> objectFiles) async {
    final requestedSymbols = _scanFfiNativeSymbols();
    final definedSymbols = await _collectDefinedSymbols([
      ...objectFiles,
      openBlasArchiveFile,
    ]);

    const allocatorSymbols = {'malloc', 'calloc', 'realloc', 'free'};
    final exportedSymbols = <String>{...allocatorSymbols};
    final missingSymbols = <String>[];
    for (final symbol in requestedSymbols) {
      if (definedSymbols.contains(symbol)) {
        exportedSymbols.add(symbol);
      } else if (!allocatorSymbols.contains(symbol)) {
        missingSymbols.add(symbol);
      }
    }

    if (missingSymbols.isNotEmpty) {
      final message =
          '${missingSymbols.length} @ffi.Native symbol(s) have no definition '
          'in the native objects or ${openBlasArchiveFile.path}: '
          '${missingSymbols.join(', ')}';
      if (!allowMissingSymbols) {
        throw _BuildException(
          '$message\nPass --allow-missing-symbols to link anyway.',
        );
      }
      stderr.writeln('Warning: $message');
    }

    final sortedExports = exportedSymbols.toList()..sort();

    final builtinsArchive = _findBuiltinsArchive();
    if (builtinsArchive == null) {
      throw const _BuildException(
        'libclang_rt.builtins.a not found in wasi-sdk.',
      );
    }

    final sysrootLibDir = '${wasiSysrootDir.path}/lib/$_wasiTargetTriple';
    final crt1Reactor = File('$sysrootLibDir/crt1-reactor.o');

    final linkArgs = <String>[
      '--no-entry',
      '--gc-sections',
      '--fatal-warnings',
      '--export=_initialize',
      for (final symbol in sortedExports) '--export=$symbol',
      crt1Reactor.path,
      '-L$sysrootLibDir/noeh',
      '-L$sysrootLibDir',
      for (final objectFile in objectFiles) objectFile.path,
      openBlasArchiveFile.path,
      '-lc++',
      '-lc++abi',
      '-lc',
      '-lm',
      '-lwasi-emulated-mman',
      '-lwasi-emulated-process-clocks',
      '-lwasi-emulated-signal',
      builtinsArchive.path,
      '-o',
      nativeWasmFile.path,
    ];

    final currentStamp = <String, Object?>{
      'schema': 1,
      'wasmLd': _wasmLdBin,
      'toolchain': _toolchainIdentity,
      'objects': {
        for (final objectFile in objectFiles)
          _baseName(objectFile.path): await _sha256OfFile(objectFile),
      },
      'openBlasArchive': await _sha256OfFile(openBlasArchiveFile),
      'exports': sortedExports,
      'linkArgs': linkArgs,
    };
    final currentStampJson = const JsonEncoder.withIndent(
      '  ',
    ).convert(currentStamp);

    if (forceRebuild) {
      stdout.writeln('Relinking ${nativeWasmFile.path}: --force-rebuild.');
    } else if (!nativeWasmFile.existsSync()) {
      stdout.writeln('Linking ${nativeWasmFile.path}: module does not exist.');
    } else if (!_linkStampFile.existsSync()) {
      stdout.writeln('Relinking ${nativeWasmFile.path}: no link stamp found.');
    } else {
      final previousStampJson = _linkStampFile.readAsStringSync();
      if (previousStampJson == currentStampJson) {
        stdout.writeln(
          'Link inputs unchanged (${_linkStampFile.path}); not relinking '
          '${nativeWasmFile.path}.',
        );
        return false;
      }
      stdout.writeln(
        'Relinking ${nativeWasmFile.path}: '
        '${_describeLinkStampChange(previousStampJson, currentStamp)}.',
      );
    }

    // Remove the stamp first so an interrupted or failed link cannot leave a
    // stale module that is mistaken for current on the next run.
    if (_linkStampFile.existsSync()) {
      _linkStampFile.deleteSync();
    }

    stdout.writeln(
      'Linking ${nativeWasmFile.path} with ${sortedExports.length} exported '
      'symbols...',
    );
    await _runChecked(_wasmLdBin, linkArgs);
    await validateNativeMathWasm();
    _linkStampFile.writeAsStringSync(currentStampJson);
    return true;
  }

  /// Describes which top-level link stamp fields differ between the stored
  /// [previousStampJson] and [currentStamp], naming individual changed
  /// objects when the object digests differ.
  static String _describeLinkStampChange(
    String previousStampJson,
    Map<String, Object?> currentStamp,
  ) {
    final Object? previous;
    try {
      previous = jsonDecode(previousStampJson);
    } on FormatException {
      return 'previous link stamp is not valid JSON';
    }
    if (previous is! Map<String, Object?>) {
      return 'previous link stamp has an unexpected shape';
    }
    final changes = <String>[];
    for (final MapEntry(key: field, value: currentValue)
        in currentStamp.entries) {
      final previousValue = previous[field];
      if (jsonEncode(previousValue) == jsonEncode(currentValue)) continue;
      if (field == 'objects' &&
          previousValue is Map<String, Object?> &&
          currentValue is Map<String, Object?>) {
        final changedObjects = [
          for (final name in {...previousValue.keys, ...currentValue.keys})
            if (previousValue[name] != currentValue[name]) name,
        ]..sort();
        changes.add('objects changed (${changedObjects.join(', ')})');
      } else {
        changes.add('$field changed');
      }
    }
    return changes.isEmpty
        ? 'link stamp formatting changed'
        : changes.join('; ');
  }

  Set<String> _scanFfiNativeSymbols() {
    final symbols = <String>{'malloc', 'calloc', 'realloc', 'free'};

    final bindingPaths = <({String path, bool Function(String) filter})>[
      (path: 'pkgs/ndarray/lib/src/ndarray_bindings.dart', filter: (_) => true),
      (
        path: 'pkgs/ndarray/lib/src/ndarray_extensions_bindings.dart',
        filter: (_) => true,
      ),
      (
        path: 'pkgs/ndarray/lib/src/operations/custom_checks.dart',
        filter: (_) => true,
      ),
      (path: 'pkgs/ndarray/lib/src/cpu_check.dart', filter: (_) => true),
      (
        path: 'pkgs/openblas/lib/src/openblas_bindings.dart',
        filter: (_) => true,
      ),
      (
        path: 'pkgs/openblas/lib/src/openblas_extensions.dart',
        filter: (_) => true,
      ),
      (
        path: 'pkgs/openblas/lib/src/openblas_extensions_bindings.dart',
        filter: (_) => true,
      ),
      (
        path: 'pkgs/pocketfft/lib/src/pocketfft_bindings.dart',
        filter: (s) =>
            s.startsWith('kiss_fft') ||
            s == 'malloc' ||
            s == 'calloc' ||
            s == 'realloc' ||
            s == 'free',
      ),
    ];

    final pattern = RegExp(
      r'''@(?:ffi\.)?Native\b[\s\S]*?(?:symbol:\s*['"]([^'"]+)['"][\s\S]*?)?\)\s*external\s+.+?\s+([a-zA-Z0-9_]+)\s*\(''',
    );

    for (final entry in bindingPaths) {
      final file = File('${repoRoot.path}/${entry.path}');
      if (!file.existsSync()) continue;
      final content = file.readAsStringSync();
      for (final match in pattern.allMatches(content)) {
        final symbol = match.group(1) ?? match.group(2)!;
        if (entry.filter(symbol)) {
          symbols.add(symbol);
        }
      }
    }

    return symbols;
  }

  Future<Set<String>> _collectDefinedSymbols(List<File> inputs) async {
    final result = await _run(_llvmNmBin, [
      '--defined-only',
      '-g',
      for (final f in inputs) f.path,
    ]);
    if (result.exitCode != 0) {
      throw _BuildException(
        'llvm-nm failed while inspecting defined symbols:\n${result.stderr}',
      );
    }
    final defined = <String>{};
    final lines = const LineSplitter().convert(result.stdout as String);
    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty || trimmed.endsWith(':')) continue;
      final parts = trimmed.split(RegExp(r'\s+'));
      if (parts.isNotEmpty) {
        defined.add(parts.last);
      }
    }
    return defined;
  }

  /// Writes `.dart_tool/wasm_build/run_wasm.mjs` to instantiate
  /// `native_math.wasm` under Node WASI and bridge `additionalImports` to Dart.
  void writeRunWasmMjs() {
    final scriptContent = _runWasmMjsTemplate
        .replaceAll('__REPO_ROOT__', jsonEncode(repoRoot.path))
        .replaceAll(
          '__EXPECT_TEST_SUMMARY_VARIABLE__',
          jsonEncode(_expectTestSummaryEnvironmentVariable),
        );

    if (!runWasmMjsFile.existsSync() ||
        runWasmMjsFile.readAsStringSync() != scriptContent) {
      runWasmMjsFile.writeAsStringSync(scriptContent);
    }
  }

  // TODO(sigurdm): This `package:ffi` overlay exists only because the
  // published package's `allocation.dart`, `utf8.dart`, and `utf16.dart` call
  // `Platform.isWindows` and `Pointer.asTypedList`, neither of which dart2wasm
  // supports. The plan is to upstream the
  // `bool.fromEnvironment('dart.tool.dart2wasm')` conditionals used in the
  // overlay sources below to `package:ffi` and then delete this overlay
  // together with `_expectedFfiPackageVersion`.

  /// Generates a Wasm-compatible overlay of `package:ffi` in
  /// `.dart_tool/wasm_build/wasm_ffi_pkg` and writes
  /// `.dart_tool/wasm_build/wasm_package_config.json` so `malloc`, `calloc`,
  /// `Utf8Pointer.toDartString`, and `StringUtf8Pointer.toNativeUtf8` work on
  /// `dart2wasm` without calling `Platform.isWindows` or `.asTypedList`.
  ///
  /// The overlay sources were derived from `package:ffi`
  /// [_expectedFfiPackageVersion]; a warning is printed when the host resolves
  /// a different version.
  void ensureWasmPackageConfig() {
    final hostPackageConfigFile = File(
      '${repoRoot.path}/.dart_tool/package_config.json',
    );
    if (!hostPackageConfigFile.existsSync()) {
      throw const _BuildException(
        'Missing .dart_tool/package_config.json. Run `dart pub get` first.',
      );
    }

    final wasmFfiDir = Directory('${buildDir.path}/wasm_ffi_pkg');
    final wasmFfiLibSourceDir = Directory('${wasmFfiDir.path}/lib/src')
      ..createSync(recursive: true);

    final decoded = jsonDecode(hostPackageConfigFile.readAsStringSync());
    if (decoded is! Map<String, Object?>) {
      throw _BuildException(
        '${hostPackageConfigFile.path} is not a JSON object.',
      );
    }
    final rawPackages = decoded['packages'];
    if (rawPackages is! List<Object?>) {
      throw _BuildException(
        '${hostPackageConfigFile.path} has no "packages" list.',
      );
    }
    final packages = <Map<String, Object?>>[];
    for (final rawPackage in rawPackages) {
      if (rawPackage is! Map<String, Object?>) {
        throw _BuildException(
          '${hostPackageConfigFile.path}: "packages" entry is not a JSON '
          'object: $rawPackage',
        );
      }
      packages.add(rawPackage);
    }

    Directory? hostFfiRoot;
    final baseConfigUri = hostPackageConfigFile.uri;

    for (final package in packages) {
      if (package['rootUri'] case final String rawRootUri) {
        final resolvedUri = baseConfigUri.resolve(rawRootUri);
        package['rootUri'] = resolvedUri.toString();
        if (package['name'] == 'ffi') {
          hostFfiRoot = Directory.fromUri(resolvedUri);
          package['rootUri'] = wasmFfiDir.uri.toString();
        }
      }
    }

    if (hostFfiRoot == null || !hostFfiRoot.existsSync()) {
      throw _BuildException(
        'package:ffi is not resolved in ${hostPackageConfigFile.path}; the '
        'Wasm overlay cannot be generated. Run `dart pub get` first.',
      );
    }

    final hostPubspec = File('${hostFfiRoot.path}/pubspec.yaml');
    if (hostPubspec.existsSync()) {
      hostPubspec.copySync('${wasmFfiDir.path}/pubspec.yaml');
      final hostFfiVersion = RegExp(
        r'^version:\s*(\S+)',
        multiLine: true,
      ).firstMatch(hostPubspec.readAsStringSync())?.group(1);
      if (hostFfiVersion != _expectedFfiPackageVersion) {
        stderr.writeln(
          '!!! WARNING: the resolved package:ffi is version '
          '${hostFfiVersion ?? '<unknown>'} but the Wasm overlay in '
          'tool/build_wasm.dart was derived from $_expectedFfiPackageVersion. '
          'Review the overlay sources against ${hostFfiRoot.path}/lib/src.',
        );
      }
    }
    final hostFfiEntry = File('${hostFfiRoot.path}/lib/ffi.dart');
    if (hostFfiEntry.existsSync()) {
      hostFfiEntry.copySync('${wasmFfiDir.path}/lib/ffi.dart');
    }
    final hostArena = File('${hostFfiRoot.path}/lib/src/arena.dart');
    if (hostArena.existsSync()) {
      hostArena.copySync('${wasmFfiLibSourceDir.path}/arena.dart');
    }

    File('${wasmFfiLibSourceDir.path}/allocation.dart').writeAsStringSync('''
import 'dart:ffi';
import 'dart:io';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');
bool get _isWindows => !_isWasm && Platform.isWindows;

typedef PosixMallocNative = Pointer Function(IntPtr);

@Native<PosixMallocNative>(symbol: 'malloc')
external Pointer posixMalloc(int size);

typedef PosixCallocNative = Pointer Function(IntPtr num, IntPtr size);

@Native<PosixCallocNative>(symbol: 'calloc')
external Pointer posixCalloc(int num, int size);

typedef PosixFreeNative = Void Function(Pointer);

@Native<Void Function(Pointer)>(symbol: 'free')
external void posixFree(Pointer ptr);

final Pointer<NativeFunction<PosixFreeNative>> posixFreePointer =
    Native.addressOf(posixFree);

final DynamicLibrary ole32lib = DynamicLibrary.open('ole32.dll');

typedef WinCoTaskMemAllocNative = Pointer Function(Size);
typedef WinCoTaskMemAlloc = Pointer Function(int);
final WinCoTaskMemAlloc winCoTaskMemAlloc = ole32lib
    .lookupFunction<WinCoTaskMemAllocNative, WinCoTaskMemAlloc>(
      'CoTaskMemAlloc',
    );

typedef WinCoTaskMemFreeNative = Void Function(Pointer);
typedef WinCoTaskMemFree = void Function(Pointer);
final Pointer<NativeFunction<WinCoTaskMemFreeNative>> winCoTaskMemFreePointer =
    ole32lib.lookup('CoTaskMemFree');
final WinCoTaskMemFree winCoTaskMemFree = winCoTaskMemFreePointer.asFunction();

final class MallocAllocator implements Allocator {
  const MallocAllocator._();

  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    if (byteCount < 0 || (_isWasm && byteCount > 0x7fffffff)) {
      throw ArgumentError('Could not allocate \$byteCount bytes.');
    }
    Pointer<T> result;
    if (_isWindows) {
      result = winCoTaskMemAlloc(byteCount).cast();
    } else {
      result = posixMalloc(byteCount).cast();
    }
    if (result.address == 0) {
      throw ArgumentError('Could not allocate \$byteCount bytes.');
    }
    return result;
  }

  @override
  void free(Pointer pointer) {
    if (_isWindows) {
      winCoTaskMemFree(pointer);
    } else {
      posixFree(pointer);
    }
  }

  Pointer<NativeFinalizerFunction> get nativeFree =>
      _isWindows ? winCoTaskMemFreePointer : posixFreePointer;
}

const MallocAllocator malloc = MallocAllocator._();

final class CallocAllocator implements Allocator {
  const CallocAllocator._();

  void _fillMemory(Pointer destination, int length, int fill) {
    final ptr = destination.cast<Uint8>();
    for (var i = 0; i < length; i++) {
      ptr[i] = fill;
    }
  }

  void _zeroMemory(Pointer destination, int length) =>
      _fillMemory(destination, length, 0);

  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    if (byteCount < 0 || (_isWasm && byteCount > 0x7fffffff)) {
      throw ArgumentError('Could not allocate \$byteCount bytes.');
    }
    Pointer<T> result;
    if (_isWindows) {
      result = winCoTaskMemAlloc(byteCount).cast();
    } else {
      result = posixCalloc(byteCount, 1).cast();
    }
    if (result.address == 0) {
      throw ArgumentError('Could not allocate \$byteCount bytes.');
    }
    if (_isWindows) {
      _zeroMemory(result, byteCount);
    }
    return result;
  }

  @override
  void free(Pointer pointer) {
    if (_isWindows) {
      winCoTaskMemFree(pointer);
    } else {
      posixFree(pointer);
    }
  }

  Pointer<NativeFinalizerFunction> get nativeFree =>
      _isWindows ? winCoTaskMemFreePointer : posixFreePointer;
}

const CallocAllocator calloc = CallocAllocator._();
''');

    File('${wasmFfiLibSourceDir.path}/utf8.dart').writeAsStringSync('''
import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import '../ffi.dart';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

final class Utf8 extends Opaque {}

extension Utf8Pointer on Pointer<Utf8> {
  int get length {
    _ensureNotNullptr('length');
    final codeUnits = cast<Uint8>();
    return _length(codeUnits);
  }

  String toDartString({int? length}) {
    _ensureNotNullptr('toDartString');
    final codeUnits = cast<Uint8>();
    if (length != null) {
      RangeError.checkNotNegative(length, 'length');
    } else {
      length = _length(codeUnits);
    }
    if (_isWasm) {
      final bytes = Uint8List(length);
      for (var i = 0; i < length; i++) {
        bytes[i] = codeUnits[i];
      }
      return utf8.decode(bytes);
    }
    return utf8.decode(codeUnits.asTypedList(length));
  }

  static int _length(Pointer<Uint8> codeUnits) {
    var length = 0;
    while (codeUnits[length] != 0) {
      length++;
    }
    return length;
  }

  void _ensureNotNullptr(String operation) {
    if (this == nullptr) {
      throw UnsupportedError(
        "Operation '\$operation' not allowed on a 'nullptr'.",
      );
    }
  }
}

extension StringUtf8Pointer on String {
  Pointer<Utf8> toNativeUtf8({Allocator allocator = malloc}) {
    final units = utf8.encode(this);
    final result = allocator<Uint8>(units.length + 1);
    if (_isWasm) {
      for (var i = 0; i < units.length; i++) {
        result[i] = units[i];
      }
      result[units.length] = 0;
    } else {
      final nativeString = result.asTypedList(units.length + 1);
      nativeString.setAll(0, units);
      nativeString[units.length] = 0;
    }
    return result.cast();
  }
}
''');

    File('${wasmFfiLibSourceDir.path}/utf16.dart').writeAsStringSync('''
import 'dart:ffi';
import 'dart:typed_data';

import '../ffi.dart';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

final class Utf16 extends Opaque {}

extension Utf16Pointer on Pointer<Utf16> {
  int get length {
    _ensureNotNullptr('length');
    final codeUnits = cast<Uint16>();
    return _length(codeUnits);
  }

  String toDartString({int? length}) {
    _ensureNotNullptr('toDartString');
    final codeUnits = cast<Uint16>();
    if (length == null) {
      return _toUnknownLengthString(codeUnits);
    } else {
      RangeError.checkNotNegative(length, 'length');
      return _toKnownLengthString(codeUnits, length);
    }
  }

  static String _toKnownLengthString(Pointer<Uint16> codeUnits, int length) {
    if (_isWasm) {
      final units = Uint16List(length);
      for (var i = 0; i < length; i++) {
        units[i] = codeUnits[i];
      }
      return String.fromCharCodes(units);
    }
    return String.fromCharCodes(codeUnits.asTypedList(length));
  }

  static String _toUnknownLengthString(Pointer<Uint16> codeUnits) {
    final buffer = StringBuffer();
    var i = 0;
    while (true) {
      final char = (codeUnits + i).value;
      if (char == 0) {
        return buffer.toString();
      }
      buffer.writeCharCode(char);
      i++;
    }
  }

  static int _length(Pointer<Uint16> codeUnits) {
    var length = 0;
    while (codeUnits[length] != 0) {
      length++;
    }
    return length;
  }

  void _ensureNotNullptr(String operation) {
    if (this == nullptr) {
      throw UnsupportedError(
        "Operation '\$operation' not allowed on a 'nullptr'.",
      );
    }
  }
}

extension StringUtf16Pointer on String {
  Pointer<Utf16> toNativeUtf16({Allocator allocator = malloc}) {
    final units = codeUnits;
    final result = allocator<Uint16>(units.length + 1);
    if (_isWasm) {
      for (var i = 0; i < units.length; i++) {
        result[i] = units[i];
      }
      result[units.length] = 0;
    } else {
      final nativeString = result.asTypedList(units.length + 1);
      nativeString.setRange(0, units.length, units);
      nativeString[units.length] = 0;
    }
    return result.cast();
  }
}
''');

    wasmPackageConfigFile.writeAsStringSync(
      const JsonEncoder.withIndent('  ').convert(decoded),
    );
  }

  /// Verifies that V8 (`node`) compiles `native_math.wasm` without any Wasm
  /// bytecode validation or stack-height errors and exports the required
  /// allocator and reactor symbols.
  Future<void> validateNativeMathWasm() async {
    await _runChecked('node', [
      '-e',
      'const fs = require("node:fs"); '
          'const bytes = fs.readFileSync(${jsonEncode(nativeWasmFile.path)}); '
          'const mod = new WebAssembly.Module(bytes); '
          'const exports = new Set(WebAssembly.Module.exports(mod).map((e) => e.name)); '
          'for (const req of ["memory", "malloc", "calloc", "realloc", "free", "_initialize"]) { '
          '  if (!exports.has(req)) throw new Error("native_math.wasm missing export: " + req); '
          '}',
    ]);
  }

  /// Discovers all Wasm-compatible `*_test.dart` files across `pkgs/pocketfft`,
  /// `pkgs/openblas`, and `pkgs/ndarray`.
  ///
  /// Throws a [_BuildException] if any [excludedWasmTests] entry does not
  /// exist on disk, so the exclusion list cannot silently go stale.
  List<String> discoverWasmTests() {
    final missingExclusions = [
      for (final path in excludedWasmTests)
        if (!File('${repoRoot.path}/$path').existsSync()) path,
    ];
    if (missingExclusions.isNotEmpty) {
      throw _BuildException(
        'excludedWasmTests lists test files that do not exist (remove or '
        'update these entries in tool/build_wasm.dart):\n  '
        '${missingExclusions.join('\n  ')}',
      );
    }

    const testRoots = [
      'pkgs/pocketfft/test',
      'pkgs/openblas/test',
      'pkgs/ndarray/test',
    ];
    final discovered = <String>[];
    for (final relativeDirectory in testRoots) {
      final dir = Directory('${repoRoot.path}/$relativeDirectory');
      if (!dir.existsSync()) continue;
      final files =
          dir
              .listSync(recursive: true, followLinks: false)
              .whereType<File>()
              .where((f) => f.path.endsWith('_test.dart'))
              .toList()
            ..sort((a, b) => a.path.compareTo(b.path));
      for (final file in files) {
        final relativePath = _normalizeRepoRelativePath(repoRoot, file.path);
        if (!excludedWasmTests.contains(relativePath)) {
          discovered.add(relativePath);
        }
      }
    }
    return discovered;
  }

  /// Compiles each Dart file in [targetPaths] (workspace-relative, or absolute
  /// for files outside the workspace) via `dart compile wasm` and runs it with
  /// Node.js + `run_wasm.mjs`, at most [testJobs] targets at a time.
  ///
  /// Each compile and each run is bounded by [testTimeout]. For `_test.dart`
  /// targets the loader is told (via [_expectTestSummaryEnvironmentVariable])
  /// to require a `package:test` summary line, so a test file whose `main`
  /// runs no tests is reported as failed.
  ///
  /// Every target is compiled with `-D[_testTemporaryDirectoryDefine]=<dir>`
  /// pointing at a fresh, target-private directory below
  /// `.dart_tool/wasm_build/test_tmp` (inside the workspace, which the loader
  /// preopens); the directory is deleted again once the run has finished.
  ///
  /// Returns `true` if all targets succeeded, or `false` if any failed.
  Future<bool> runDartTargets(List<String> targetPaths) async {
    if (targetPaths.isEmpty) {
      stdout.writeln('No Dart test targets matched.');
      return true;
    }

    final testOutputRoot = Directory('${buildDir.path}/test_out');
    testOutputRoot.createSync(recursive: true);
    final testTemporaryRoot = Directory('${buildDir.path}/test_tmp');
    testTemporaryRoot.createSync(recursive: true);

    stdout.writeln(
      'Running ${targetPaths.length} Wasm target(s) with concurrency '
      '$testJobs and a ${testTimeout.inSeconds}s timeout per stage...',
    );

    var passedCount = 0;
    final failures = <({String path, String stage, String output})>[];

    await _runPool<String>(targetPaths, testJobs, (targetPath) async {
      final slug = targetPath
          .replaceAll('/', '__')
          .replaceAll(RegExp(r'\.dart$'), '');
      final outputDir = Directory('${testOutputRoot.path}/$slug')
        ..createSync(recursive: true);
      final wasmOutput = File('${outputDir.path}/$slug.wasm');
      final mjsOutput = File('${outputDir.path}/$slug.mjs');

      final sanitizedTarget = targetPath.replaceAll(
        RegExp(r'[^A-Za-z0-9._-]'),
        '_',
      );
      final scratchDir = Directory(
        '${testTemporaryRoot.path}/$sanitizedTarget',
      );
      if (scratchDir.existsSync()) {
        scratchDir.deleteSync(recursive: true);
      }
      scratchDir.createSync(recursive: true);

      try {
        final compileResult = await _runWithTimeout(
          Platform.resolvedExecutable,
          [
            'compile',
            'wasm',
            '--packages=${wasmPackageConfigFile.path}',
            '--extra-compiler-option=--enable-experimental-ffi',
            '-D$_testTemporaryDirectoryDefine=${scratchDir.path}',
            '-O1',
            targetPath,
            '-o',
            wasmOutput.path,
          ],
          workingDirectory: repoRoot.path,
          timeout: testTimeout,
        );

        if (compileResult.timedOut) {
          failures.add((
            path: targetPath,
            stage:
                'timeout (dart compile wasm exceeded '
                '${testTimeout.inSeconds}s)',
            output: compileResult.combinedOutput,
          ));
          stderr.writeln('[FAIL] $targetPath (compile timeout)');
          return;
        }
        if (compileResult.exitCode != 0) {
          failures.add((
            path: targetPath,
            stage: 'dart compile wasm (exit ${compileResult.exitCode})',
            output: compileResult.combinedOutput,
          ));
          stderr.writeln('[FAIL] $targetPath (compile error)');
          return;
        }

        final isTestFile = targetPath.endsWith('_test.dart');
        final runResult = await _runWithTimeout(
          'node',
          [
            '--experimental-wasi-unstable-preview1',
            runWasmMjsFile.path,
            wasmOutput.path,
            mjsOutput.path,
          ],
          workingDirectory: repoRoot.path,
          environment: {
            ...Platform.environment,
            _expectTestSummaryEnvironmentVariable: isTestFile ? '1' : '0',
          },
          timeout: testTimeout,
        );

        if (runResult.timedOut) {
          failures.add((
            path: targetPath,
            stage: 'timeout (node run exceeded ${testTimeout.inSeconds}s)',
            output: runResult.combinedOutput,
          ));
          stderr.writeln('[FAIL] $targetPath (run timeout)');
          return;
        }
        if (runResult.exitCode != 0) {
          failures.add((
            path: targetPath,
            stage: 'node run_wasm.mjs (exit ${runResult.exitCode})',
            output: runResult.combinedOutput,
          ));
          stderr.writeln('[FAIL] $targetPath');
          return;
        }

        passedCount++;
        stdout.writeln('[PASS] $targetPath');
      } finally {
        try {
          if (scratchDir.existsSync()) {
            scratchDir.deleteSync(recursive: true);
          }
        } on FileSystemException {
          // Best effort: a leftover scratch directory must not fail the run.
        }
      }
    });

    stdout.writeln();
    stdout.writeln(
      'Wasm Summary: $passedCount passed, ${failures.length} failed '
      '(out of ${targetPaths.length}).',
    );

    if (failures.isNotEmpty) {
      for (final failure in failures) {
        stderr.writeln('--- FAILURE: ${failure.path} (${failure.stage}) ---');
        stderr.writeln(failure.output);
      }
      return false;
    }
    return true;
  }

  /// Runs [executable] to completion while draining its output, killing it
  /// (SIGTERM, then SIGKILL after a grace period) once [timeout] elapses.
  ///
  /// Throws a [_BuildException] if the process cannot be started.
  Future<_TimedProcessResult> _runWithTimeout(
    String executable,
    List<String> arguments, {
    required Duration timeout,
    String? workingDirectory,
    Map<String, String>? environment,
  }) async {
    final Process process;
    try {
      process = await Process.start(
        executable,
        arguments,
        workingDirectory: workingDirectory,
        environment: environment,
      );
    } on ProcessException catch (e) {
      throw _BuildException(
        '$executable not found or failed to start: ${e.message}',
      );
    }

    const decoder = Utf8Decoder(allowMalformed: true);
    final stdoutBuffer = StringBuffer();
    final stderrBuffer = StringBuffer();
    final stdoutDone = process.stdout
        .transform(decoder)
        .forEach(stdoutBuffer.write);
    final stderrDone = process.stderr
        .transform(decoder)
        .forEach(stderrBuffer.write);

    var timedOut = false;
    Timer? killTimer;
    final timeoutTimer = Timer(timeout, () {
      timedOut = true;
      process.kill(ProcessSignal.sigterm);
      killTimer = Timer(const Duration(seconds: 5), () {
        process.kill(ProcessSignal.sigkill);
      });
    });

    final exitCode = await process.exitCode;
    timeoutTimer.cancel();
    killTimer?.cancel();
    // Grandchildren that inherited the pipes could keep them open after the
    // child was killed; do not wait for them indefinitely.
    await Future.wait([
      stdoutDone,
      stderrDone,
    ]).timeout(const Duration(seconds: 10), onTimeout: () => const []);

    return _TimedProcessResult(
      exitCode: exitCode,
      stdout: stdoutBuffer.toString(),
      stderr: stderrBuffer.toString(),
      timedOut: timedOut,
    );
  }

  /// Runs [executable] to completion, converting a failure to start it into a
  /// [_BuildException].
  Future<ProcessResult> _run(
    String executable,
    List<String> arguments, {
    String? workingDirectory,
    Map<String, String>? environment,
  }) async {
    try {
      return await Process.run(
        executable,
        arguments,
        workingDirectory: workingDirectory,
        environment: environment,
      );
    } on ProcessException catch (e) {
      throw _BuildException(
        '$executable not found or failed to start: ${e.message}',
      );
    }
  }

  /// Runs [executable] and throws a [_BuildException] with its output on a
  /// non-zero exit code.
  ///
  /// When [echoOutput] is set, non-empty (trimmed) stdout and stderr of a
  /// successful run are forwarded so tool warnings (e.g. from `wasm-ld`)
  /// remain visible.
  Future<void> _runChecked(
    String executable,
    List<String> arguments, {
    String? workingDirectory,
    Map<String, String>? environment,
    bool echoOutput = true,
  }) async {
    final result = await _run(
      executable,
      arguments,
      workingDirectory: workingDirectory,
      environment: environment,
    );
    if (result.exitCode != 0) {
      throw _BuildException(
        'Command failed (exit ${result.exitCode}): '
        '$executable ${arguments.join(' ')}\n'
        '${result.stdout}\n${result.stderr}',
      );
    }
    if (!echoOutput) return;
    final standardOutput = '${result.stdout}'.trim();
    final standardError = '${result.stderr}'.trim();
    if (standardOutput.isNotEmpty) {
      stdout.writeln(standardOutput);
    }
    if (standardError.isNotEmpty) {
      stderr.writeln('${_baseName(executable)}: $standardError');
    }
  }
}

/// Template of `run_wasm.mjs`; `__REPO_ROOT__` and
/// `__EXPECT_TEST_SUMMARY_VARIABLE__` are replaced with JSON string literals
/// by [_WasmWorkspaceBuilder.writeRunWasmMjs].
const String _runWasmMjsTemplate = r'''
// Generated by tool/build_wasm.dart; do not edit.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { WASI } from 'node:wasi';

const repoRoot = __REPO_ROOT__;
const expectTestSummaryVariable = __EXPECT_TEST_SUMMARY_VARIABLE__;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const [dartWasmPath, dartMjsPath, ...testArgs] = process.argv.slice(2);
if (!dartWasmPath || !dartMjsPath) {
  console.error(
    'Usage: node --experimental-wasi-unstable-preview1 run_wasm.mjs <dart.wasm> <dart.mjs> [args...]',
  );
  process.exit(2);
}

// The sandbox only sees a filtered environment and a minimal set of
// preopened directories: tests write below /tmp and read fixtures below the
// workspace root.
const allowedEnvironmentNames = new Set(['PATH', 'HOME', 'TMPDIR']);
const allowedEnvironmentPrefixes = ['NDARRAY_', 'OPENBLAS_', 'OMP_'];
const sandboxEnvironment = {};
for (const [name, value] of Object.entries(process.env)) {
  if (
    name === expectTestSummaryVariable ||
    allowedEnvironmentNames.has(name) ||
    allowedEnvironmentPrefixes.some((prefix) => name.startsWith(prefix))
  ) {
    sandboxEnvironment[name] = value;
  }
}
const expectTestSummary = process.env[expectTestSummaryVariable] === '1';

const wasi = new WASI({
  version: 'preview1',
  args: ['native_math.wasm'],
  env: sandboxEnvironment,
  preopens: {
    '/tmp': '/tmp',
    [repoRoot]: repoRoot,
    '.': process.cwd(),
  },
});

const nativeWasmPath = path.join(__dirname, 'native_math.wasm');
const nativeBytes = fs.readFileSync(nativeWasmPath);
const nativeModule = new WebAssembly.Module(nativeBytes);
const nativeInstance = new WebAssembly.Instance(nativeModule, {
  wasi_snapshot_preview1: wasi.wasiImport,
});

if (typeof nativeInstance.exports._initialize === 'function') {
  wasi.initialize(nativeInstance);
}

const nativeExports = nativeInstance.exports;

const ffiNamespace = new Proxy(nativeExports, {
  get(target, prop) {
    if (prop in target) {
      return target[prop];
    }
    throw new LinkError(
      `Unresolved @ffi.Native symbol imported by dart2wasm: ffi.${String(prop)}`,
    );
  },
});

const additionalImports = new Proxy(
  {
    ffi: ffiNamespace,
    'package:ndarray/ndarray': ffiNamespace,
    'package:ndarray/ndarray_cpu_check': ffiNamespace,
    'package:pocketfft/pocketfft': ffiNamespace,
    'package:openblas/openblas': ffiNamespace,
    'package:openblas/openblas_extensions': ffiNamespace,
    memory: {
      memory: nativeExports.memory,
      malloc: nativeExports.malloc,
      calloc: nativeExports.calloc,
      realloc: nativeExports.realloc,
      free: nativeExports.free,
    },
    wasi_snapshot_preview1: wasi.wasiImport,
  },
  {
    get(target, namespace) {
      if (namespace in target) {
        return target[namespace];
      }
      return ffiNamespace;
    },
  },
);

const dartMjsUrl = pathToFileURL(path.resolve(dartMjsPath)).href;
const dartSupport = await import(dartMjsUrl);
const dartBytes = fs.readFileSync(path.resolve(dartWasmPath));

globalThis.location ??= {
  href: pathToFileURL(path.resolve(process.cwd()) + path.sep).href,
};

// package:test, when a test file's main() is run directly, prints an expanded
// reporter whose last line is one of "All tests passed!", "All tests
// skipped.", "Some tests failed.", or "No tests ran.".
let sawTestFailure = false;
let sawTestSummary = false;
globalThis.dartPrint = (line) => {
  const text = String(line);
  console.log(text);
  if (
    text.includes('Some tests failed.') ||
    text.includes('No tests ran.') ||
    text.includes('[E]') ||
    text.includes('Unhandled exception:')
  ) {
    sawTestFailure = true;
  }
  if (text.includes('All tests passed!') || text.includes('All tests skipped.')) {
    sawTestSummary = true;
  }
};

process.on('beforeExit', () => {
  if (sawTestFailure) {
    process.exitCode = 1;
  } else if (expectTestSummary && !sawTestSummary) {
    console.error(
      'run_wasm.mjs: the test printed neither "All tests passed!" nor ' +
        '"All tests skipped."; reporting failure.',
    );
    process.exitCode = 1;
  }
});

const compiledApp = await dartSupport.compile(dartBytes);
const dartInstance = await compiledApp.instantiate(additionalImports);
dartInstance.invokeMain(...testArgs);
''';

/// One C/C++ source compiled to `objs/<objectName>` with [flags].
final class _TranslationUnit {
  const _TranslationUnit({
    required this.sourceFile,
    required this.objectName,
    required this.isCpp,
    required this.flags,
  });

  final File sourceFile;
  final String objectName;
  final bool isCpp;
  final List<String> flags;
}

/// Per-object entry of `objs/hashes.json`.
final class _ObjectCacheEntry {
  const _ObjectCacheEntry({
    required this.digest,
    required this.headers,
    required this.headersDigest,
  });

  /// Digest of the cache schema, toolchain, compile flags, and source contents.
  final String digest;

  /// Sorted absolute paths of every header the compiler read (per its depfile).
  final List<String> headers;

  /// Digest over [headers] and their contents at compile time.
  final String headersDigest;

  /// Parses a JSON object written by [toJson]; returns `null` for any other
  /// shape (including entries written by older schema versions).
  static _ObjectCacheEntry? tryParse(Object? json) {
    if (json is! Map<String, Object?>) return null;
    final digest = json['digest'];
    final headers = json['headers'];
    final headersDigest = json['headersDigest'];
    if (digest is! String ||
        headers is! List<Object?> ||
        headersDigest is! String) {
      return null;
    }
    final headerPaths = <String>[];
    for (final header in headers) {
      if (header is! String) return null;
      headerPaths.add(header);
    }
    return _ObjectCacheEntry(
      digest: digest,
      headers: headerPaths,
      headersDigest: headersDigest,
    );
  }

  Map<String, Object?> toJson() => {
    'digest': digest,
    'headers': headers,
    'headersDigest': headersDigest,
  };
}

/// Outcome of [_WasmWorkspaceBuilder._runWithTimeout].
final class _TimedProcessResult {
  const _TimedProcessResult({
    required this.exitCode,
    required this.stdout,
    required this.stderr,
    required this.timedOut,
  });

  final int exitCode;
  final String stdout;
  final String stderr;

  /// Whether the process was killed because the timeout elapsed.
  final bool timedOut;

  String get combinedOutput => '$stdout\n$stderr'.trim();
}

/// Fatal build error; reported to the user without a stack trace.
final class _BuildException implements Exception {
  const _BuildException(this.message);
  final String message;

  @override
  String toString() => 'BuildException: $message';
}

Future<void> _runPool<T>(
  List<T> items,
  int concurrency,
  Future<void> Function(T item) action,
) async {
  var index = 0;
  final workers = List.generate(
    concurrency < items.length ? concurrency : items.length,
    (_) async {
      while (true) {
        final current = index++;
        if (current >= items.length) break;
        await action(items[current]);
      }
    },
  );
  await Future.wait(workers);
}
