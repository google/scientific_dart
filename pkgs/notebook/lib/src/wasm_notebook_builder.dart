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
import 'package:path/path.dart' as p;

/// Result of building a static serverless Wasm notebook bundle.
final class WasmNotebookBundleResult {
  /// Absolute path to the staged static directory ready to be served over HTTP.
  final String outputDir;

  /// Total number of Dart source files packed into `sources_bundle.json`.
  final int bundledSourceFileCount;

  /// Size in bytes of the compiled `compiler_worker.wasm`.
  final int compilerWorkerWasmBytes;

  /// Size in bytes of `native_math.wasm`.
  final int nativeMathWasmBytes;

  /// Whether DartPad LSP worker assets were also staged under `dartpad/`.
  final bool hasDartPadWorkerAssets;

  /// Constructs a [WasmNotebookBundleResult].
  const WasmNotebookBundleResult({
    required this.outputDir,
    required this.bundledSourceFileCount,
    required this.compilerWorkerWasmBytes,
    required this.nativeMathWasmBytes,
    required this.hasDartPadWorkerAssets,
  });
}

/// Builds a self-contained static web bundle for running `pkgs/notebook`
/// completely in the browser without an active server.
///
/// The bundle contains:
/// - `index.html`, `wasm_notebook_runtime.js`, `compiler_worker_bootstrap.js`
/// - `compiler_worker.wasm` and `compiler_worker.mjs` (self-hosted `dart2wasm`
///   compiled to WebAssembly with an in-memory file system and cell bundler)
/// - `dart2wasm_platform.dill` (the Dart SDK's Wasm platform summary)
/// - `sources_bundle.json` (the Wasm-compatible workspace package sources and
///   virtual `package_config.json`)
/// - `native_math.wasm` (the Wasm32-WASI native library providing OpenBLAS,
///   PocketFFT, and `ndarray` C++ kernels)
/// - `editor_client.js`, `editor_client.wasm`, `editor_client.mjs`
/// - Optional `dartpad/` worker assets (`worker.wasm`, `worker.mjs`, `sdk.tar`)
///   when available on disk.
final class WasmNotebookBuilder {
  /// Workspace root directory (containing `tool/build_wasm.dart` and `pkgs/`).
  final String workspaceRoot;

  /// Path to the Dart SDK directory (whose `bin/dart` and
  /// `lib/_internal/dart2wasm_platform.dill` are used).
  final String dartSdkPath;

  /// Optional path to a local `dart-sdk/sdk` git checkout used to extract the
  /// internal `pkg/dart2wasm`, `pkg/front_end`, `pkg/kernel`, and `pkg/vm`
  /// sources when compiling `compiler_worker.wasm` for the first time.
  final String? dartSdkRepoCheckout;

  /// Constructs a [WasmNotebookBuilder].
  WasmNotebookBuilder({
    required this.workspaceRoot,
    String? dartSdkPath,
    this.dartSdkRepoCheckout,
  }) : dartSdkPath =
           dartSdkPath ??
           Platform.environment['DART_SDK'] ??
           p.dirname(p.dirname(Platform.resolvedExecutable));

  String get _dartExecutable =>
      p.join(dartSdkPath, 'bin', Platform.isWindows ? 'dart.exe' : 'dart');

  /// Packages from `.dart_tool/wasm_build/wasm_package_config.json` whose
  /// `lib/` sources are packed into `sources_bundle.json` for in-browser
  /// cell compilation.
  static const Set<String> _bundledCellPackages = {
    'ndarray',
    'openblas',
    'pocketfft',
    'ffi',
    'resource_scope',
    'gpuarray',
    'ndarray_ma',
    'notebook',
    'meta',
    'collection',
    'typed_data',
    'vector_math',
  };

  /// Internal Dart SDK packages needed to compile `package:dart2wasm` to Wasm.
  static const List<String> _sdkInternalPackages = [
    '_fe_analyzer_shared',
    '_js_interop_checks',
    'build_integration',
    'cfg',
    'compiler',
    'dart2wasm',
    'front_end',
    'js_ast',
    'js_runtime',
    'js_shared',
    'kernel',
    'meta',
    'mmap',
    'vm',
    'wasm_builder',
  ];

  /// Stages the complete serverless Wasm notebook static bundle into [outputDir].
  Future<WasmNotebookBundleResult> buildStaticBundle({
    String? outputDir,
    bool forceRebuildCompiler = false,
  }) async {
    final targetDir = Directory(
      outputDir ?? p.join(workspaceRoot, '.dart_tool', 'wasm_notebook_bundle'),
    );
    if (!targetDir.existsSync()) {
      targetDir.createSync(recursive: true);
    }

    // 1. Ensure native_math.wasm and wasm_package_config.json exist.
    final wasmBuildDir = p.join(workspaceRoot, '.dart_tool', 'wasm_build');
    final nativeMathFile = File(p.join(wasmBuildDir, 'native_math.wasm'));
    final wasmPkgConfigFile = File(
      p.join(wasmBuildDir, 'wasm_package_config.json'),
    );
    if (forceRebuildCompiler ||
        !nativeMathFile.existsSync() ||
        !wasmPkgConfigFile.existsSync()) {
      final pkgConfigPath = p.join(
        workspaceRoot,
        '.dart_tool',
        'package_config.json',
      );
      final res = await Process.run(_dartExecutable, [
        '--disable-dart-dev',
        '--packages=$pkgConfigPath',
        'tool/build_wasm.dart',
        '--build-only',
      ], workingDirectory: workspaceRoot);
      if (res.exitCode != 0) {
        throw StateError(
          'tool/build_wasm.dart --build-only failed (exit ${res.exitCode}):\n'
          '${res.stdout}\n${res.stderr}',
        );
      }
    }
    final targetNativeMath = File(p.join(targetDir.path, 'native_math.wasm'));
    nativeMathFile.copySync(targetNativeMath.path);

    // 2. Copy dart2wasm_platform.dill from the host Dart SDK.
    final platformDillFile = File(
      p.join(dartSdkPath, 'lib', '_internal', 'dart2wasm_platform.dill'),
    );
    if (!platformDillFile.existsSync()) {
      throw StateError(
        'Missing dart2wasm_platform.dill at ${platformDillFile.path}',
      );
    }
    File(
      p.join(targetDir.path, 'dart2wasm_platform.dill'),
    ).writeAsBytesSync(platformDillFile.readAsBytesSync());

    // 3. Build sources_bundle.json from wasm_package_config.json.
    final bundledFileCount = _writeSourcesBundleJson(
      wasmPkgConfigFile: wasmPkgConfigFile,
      outputFile: File(p.join(targetDir.path, 'sources_bundle.json')),
    );

    // 4. Build or copy cached compiler_worker.wasm + compiler_worker.mjs.
    await _ensureCompilerWorkerBuilt(
      wasmBuildDir: wasmBuildDir,
      wasmPkgConfigFile: wasmPkgConfigFile,
      targetDir: targetDir,
      forceRebuild: forceRebuildCompiler,
    );

    // 5. Copy web assets (index.html, runtime JS, worker bootstrap, editor_client).
    await _stageWebAssets(targetDir);

    // 6. Stage optional DartPad LSP worker assets if available.
    final hasDartPad = _stageDartPadAssetsIfAvailable(targetDir);

    // 7. Write .nojekyll so GitHub Pages serves all static files unaltered.
    File(p.join(targetDir.path, '.nojekyll')).writeAsStringSync('');

    final compilerWasmSize = File(
      p.join(targetDir.path, 'compiler_worker.wasm'),
    ).lengthSync();

    return WasmNotebookBundleResult(
      outputDir: targetDir.path,
      bundledSourceFileCount: bundledFileCount,
      compilerWorkerWasmBytes: compilerWasmSize,
      nativeMathWasmBytes: targetNativeMath.lengthSync(),
      hasDartPadWorkerAssets: hasDartPad,
    );
  }

  int _writeSourcesBundleJson({
    required File wasmPkgConfigFile,
    required File outputFile,
  }) {
    final rawConfig =
        jsonDecode(wasmPkgConfigFile.readAsStringSync())
            as Map<String, Object?>;
    final packages = (rawConfig['packages'] as List<Object?>)
        .cast<Map<String, Object?>>();

    final virtualPackages = <Map<String, Object?>>[];
    final filesMap = <String, String>{};

    for (final pkg in packages) {
      final name = pkg['name'] as String;
      if (!_bundledCellPackages.contains(name)) continue;

      final rootUriStr = pkg['rootUri'] as String;
      final pkgUriStr = (pkg['packageUri'] as String?) ?? 'lib/';
      final langVersion = (pkg['languageVersion'] as String?) ?? '3.10';

      final rootUri = wasmPkgConfigFile.uri.resolve(rootUriStr);
      final rootDirPath = rootUri.toFilePath();
      final libDirPath = p.normalize(p.join(rootDirPath, pkgUriStr));
      final libDir = Directory(libDirPath);
      if (!libDir.existsSync()) continue;

      virtualPackages.add({
        'name': name,
        'rootUri': 'memory:///packages/$name/',
        'packageUri': 'lib/',
        'languageVersion': langVersion,
      });

      if (name == 'notebook') {
        // Only bundle Wasm-safe library files from package:notebook.
        for (final rel in const [
          'src/notebook_widgets.dart',
          'src/ipynb.dart',
          'src/wasm_cell_bundler.dart',
        ]) {
          final f = File(p.join(libDirPath, rel));
          if (f.existsSync()) {
            filesMap['/packages/notebook/lib/$rel'] = f.readAsStringSync();
          }
        }
        continue;
      }

      for (final entity in libDir.listSync(recursive: true)) {
        if (entity is! File || !entity.path.endsWith('.dart')) continue;
        final relPath = p
            .relative(entity.path, from: libDirPath)
            .replaceAll(r'\', '/');
        filesMap['/packages/$name/lib/$relPath'] = entity.readAsStringSync();
      }
    }

    final virtualPackageConfigJson = const JsonEncoder.withIndent(
      '  ',
    ).convert({'configVersion': 2, 'packages': virtualPackages});

    final bundlePayload = jsonEncode({
      'packagesJson': virtualPackageConfigJson,
      'files': filesMap,
    });
    outputFile.writeAsStringSync(bundlePayload);
    return filesMap.length;
  }

  Future<void> _ensureCompilerWorkerBuilt({
    required String wasmBuildDir,
    required File wasmPkgConfigFile,
    required Directory targetDir,
    required bool forceRebuild,
  }) async {
    final workerBuildDir = Directory(p.join(wasmBuildDir, 'compiler_worker'));
    if (!workerBuildDir.existsSync()) {
      workerBuildDir.createSync(recursive: true);
    }

    final cachedWasm = File(
      p.join(workerBuildDir.path, 'compiler_worker.wasm'),
    );
    final cachedMjs = File(p.join(workerBuildDir.path, 'compiler_worker.mjs'));
    final workerSourceFile = File(
      p.join(workerBuildDir.path, 'wasm_compiler_worker.dart'),
    );
    final stampFile = File(
      p.join(workerBuildDir.path, 'compiler_worker.stamp'),
    );

    final workerSource = _generateCompilerWorkerDartSource();
    final bundlerSource = File(
      p.join(
        workspaceRoot,
        'pkgs',
        'notebook',
        'lib',
        'src',
        'wasm_cell_bundler.dart',
      ),
    ).readAsStringSync();
    final desiredStamp = '${workerSource.length}:${bundlerSource.length}:v1';

    if (!forceRebuild &&
        cachedWasm.existsSync() &&
        cachedMjs.existsSync() &&
        stampFile.existsSync() &&
        stampFile.readAsStringSync().trim() == desiredStamp) {
      cachedWasm.copySync(p.join(targetDir.path, 'compiler_worker.wasm'));
      cachedMjs.copySync(p.join(targetDir.path, 'compiler_worker.mjs'));
      return;
    }

    final sdk314Dir = Directory(p.join(wasmBuildDir, 'sdk_3_14'));
    await _ensureSdkInternalPackagesExtracted(sdk314Dir);

    // Build package_config.json for compiling wasm_compiler_worker.dart.
    final rawConfig =
        jsonDecode(wasmPkgConfigFile.readAsStringSync())
            as Map<String, Object?>;
    final basePackages = (rawConfig['packages'] as List<Object?>)
        .cast<Map<String, Object?>>();
    final mergedByName = <String, Map<String, Object?>>{};
    for (final pkg in basePackages) {
      final name = pkg['name'] as String;
      mergedByName[name] = Map<String, Object?>.of(pkg);
    }

    for (final sdkPkgName in _sdkInternalPackages) {
      final pkgDir = Directory(p.join(sdk314Dir.path, 'pkg', sdkPkgName));
      if (!pkgDir.existsSync()) continue;
      var languageVersion = '3.13';
      final pubspecFile = File(p.join(pkgDir.path, 'pubspec.yaml'));
      if (pubspecFile.existsSync()) {
        final match = RegExp(
          r'sdk:\s*["\x27]?\^?(\d+\.\d+)',
        ).firstMatch(pubspecFile.readAsStringSync());
        if (match != null) {
          languageVersion = match.group(1)!;
        }
      }
      mergedByName[sdkPkgName] = {
        'name': sdkPkgName,
        'rootUri': p.toUri(pkgDir.path).toString(),
        'packageUri': 'lib/',
        'languageVersion': languageVersion,
      };
    }

    final workerPkgConfigFile = File(
      p.join(workerBuildDir.path, 'package_config.json'),
    );
    workerPkgConfigFile.writeAsStringSync(
      const JsonEncoder.withIndent(
        '  ',
      ).convert({'configVersion': 2, 'packages': mergedByName.values.toList()}),
    );

    workerSourceFile.writeAsStringSync(workerSource);

    final compileRes = await Process.run(_dartExecutable, [
      'compile',
      'wasm',
      '--packages=${workerPkgConfigFile.path}',
      '-O1',
      '--no-source-maps',
      workerSourceFile.path,
      '-o',
      cachedWasm.path,
    ], workingDirectory: workspaceRoot);
    if (compileRes.exitCode != 0) {
      throw StateError(
        'Failed to compile wasm_compiler_worker.dart (exit ${compileRes.exitCode}):\n'
        '${compileRes.stdout}\n${compileRes.stderr}',
      );
    }

    stampFile.writeAsStringSync(desiredStamp);
    cachedWasm.copySync(p.join(targetDir.path, 'compiler_worker.wasm'));
    cachedMjs.copySync(p.join(targetDir.path, 'compiler_worker.mjs'));
  }

  Future<void> _ensureSdkInternalPackagesExtracted(Directory sdk314Dir) async {
    final dart2wasmDir = Directory(
      p.join(sdk314Dir.path, 'pkg', 'dart2wasm', 'lib'),
    );
    final cfgDir = Directory(p.join(sdk314Dir.path, 'pkg', 'cfg', 'lib'));
    if (!dart2wasmDir.existsSync() || !cfgDir.existsSync()) {
      final candidates = <String>[
        ?dartSdkRepoCheckout,
        ?Platform.environment['DART_SDK_REPO'],
        '/usr/local/google/home/sigurdm/projects/dart-sdk/sdk',
      ];

      String? gitDir;
      for (final candidate in candidates) {
        final dotGit = Directory(p.join(candidate, '.git'));
        if (dotGit.existsSync()) {
          gitDir = dotGit.path;
          break;
        }
      }

      if (!sdk314Dir.existsSync()) {
        sdk314Dir.createSync(recursive: true);
      }

      // Determine the SDK version tag (e.g. 3.14.0-142.0.dev) from `dart --version`.
      final verRes = await Process.run(_dartExecutable, ['--version']);
      final verText = '${verRes.stdout}\n${verRes.stderr}';
      final verMatch = RegExp(
        r'(\d+\.\d+\.\d+(?:-[\w.]+)?)',
      ).firstMatch(verText);
      final ref = verMatch?.group(1) ?? '3.14.0-142.0.dev';
      final pkgPaths = _sdkInternalPackages.map((n) => 'pkg/$n').toList();

      if (gitDir != null) {
        final archiveRes = await Process.run('bash', [
          '-c',
          'git --git-dir="$gitDir" archive "$ref" ${pkgPaths.join(' ')} | tar -x -C "${sdk314Dir.path}"',
        ]);
        if (archiveRes.exitCode != 0) {
          throw StateError(
            'Failed to extract SDK internal packages at $ref:\n'
            '${archiveRes.stdout}\n${archiveRes.stderr}',
          );
        }
      } else {
        // Fallback for CI / machines without a local dart-sdk git checkout:
        // perform a blobless shallow sparse checkout from github.com/dart-lang/sdk.
        final tmpCloneDir = Directory.systemTemp.createTempSync(
          'dart_sdk_sparse_',
        );
        try {
          var cloneRes = await Process.run('git', [
            'clone',
            '--depth',
            '1',
            '--branch',
            ref,
            '--filter=blob:none',
            '--sparse',
            'https://github.com/dart-lang/sdk.git',
            tmpCloneDir.path,
          ]);
          if (cloneRes.exitCode != 0) {
            cloneRes = await Process.run('git', [
              'clone',
              '--depth',
              '1',
              '--filter=blob:none',
              '--sparse',
              'https://github.com/dart-lang/sdk.git',
              tmpCloneDir.path,
            ]);
          }
          if (cloneRes.exitCode != 0) {
            throw StateError(
              'Failed to sparse-clone dart-lang/sdk ($ref):\n'
              '${cloneRes.stdout}\n${cloneRes.stderr}',
            );
          }
          final sparseRes = await Process.run('git', [
            '-C',
            tmpCloneDir.path,
            'sparse-checkout',
            'set',
            ...pkgPaths,
          ]);
          if (sparseRes.exitCode != 0) {
            throw StateError(
              'git sparse-checkout failed:\n'
              '${sparseRes.stdout}\n${sparseRes.stderr}',
            );
          }
          for (final pkgName in _sdkInternalPackages) {
            final srcPkg = Directory(p.join(tmpCloneDir.path, 'pkg', pkgName));
            if (!srcPkg.existsSync()) continue;
            final dstPkg = Directory(p.join(sdk314Dir.path, 'pkg', pkgName));
            dstPkg.parent.createSync(recursive: true);
            final cpRes = await Process.run('cp', [
              '-r',
              srcPkg.path,
              dstPkg.parent.path,
            ]);
            if (cpRes.exitCode != 0) {
              throw StateError('Failed to copy pkg/$pkgName: ${cpRes.stderr}');
            }
          }
        } finally {
          try {
            tmpCloneDir.deleteSync(recursive: true);
          } catch (_) {}
        }
      }
    }

    // Stub out CLI-only `--dry-run` (which otherwise pulls in package:analyzer
    // and package:linter) and Dart2JsBuilder in program_split_constraints
    // (which otherwise pulls in the entire dart2js compiler).
    final dryRunFile = File(
      p.join(sdk314Dir.path, 'pkg', 'dart2wasm', 'lib', 'dry_run.dart'),
    );
    if (dryRunFile.existsSync()) {
      dryRunFile.writeAsStringSync('''
import 'package:kernel/ast.dart';

class DryRunSummarizer {
  final Component component;
  final bool enableExperimentalFfi;
  DryRunSummarizer(this.component, {this.enableExperimentalFfi = false});
  Future<bool> summarize() async => false;
}
''');
    }

    final splitBuilderFile = File(
      p.join(
        sdk314Dir.path,
        'pkg',
        'compiler',
        'lib',
        'src',
        'deferred_load',
        'program_split_constraints',
        'builder.dart',
      ),
    );
    if (splitBuilderFile.existsSync()) {
      final src = splitBuilderFile.readAsStringSync();
      if (src.contains("import '../../elements/entities.dart';")) {
        final patched = src
            .replaceFirst("import '../../elements/entities.dart';\n", '')
            .replaceFirst(
              RegExp(
                r'class Dart2JsBuilder extends _Builder<ImportEntity>\s*\{[\s\S]*?\n\}\n',
              ),
              '',
            );
        splitBuilderFile.writeAsStringSync(patched);
      }
    }
  }

  Future<void> _stageWebAssets(Directory targetDir) async {
    final notebookDir = p.join(workspaceRoot, 'pkgs', 'notebook');
    final webDir = Directory(p.join(notebookDir, 'web'));

    // Ensure editor_client.wasm is compiled and up to date with its sources.
    final editorWasm = File(p.join(webDir.path, 'editor_client.wasm'));
    final editorMjs = File(p.join(webDir.path, 'editor_client.mjs'));
    final editorDart = File(p.join(webDir.path, 'editor_client.dart'));
    final cellFormatterDart = File(
      p.join(notebookDir, 'lib', 'src', 'cell_formatter.dart'),
    );
    var needsCompile = !editorWasm.existsSync() || !editorMjs.existsSync();
    if (!needsCompile && editorDart.existsSync()) {
      final wasmModified = editorWasm.lastModifiedSync();
      if (editorDart.lastModifiedSync().isAfter(wasmModified) ||
          (cellFormatterDart.existsSync() &&
              cellFormatterDart.lastModifiedSync().isAfter(wasmModified))) {
        needsCompile = true;
      }
    }
    if (needsCompile && editorDart.existsSync()) {
      final pkgConfig = File(
        p.join(workspaceRoot, '.dart_tool', 'package_config.json'),
      );
      await Process.run(_dartExecutable, [
        'compile',
        'wasm',
        if (pkgConfig.existsSync()) '--packages=${pkgConfig.path}',
        editorDart.path,
        '-o',
        editorWasm.path,
      ], workingDirectory: notebookDir);
    }

    for (final fileName in const [
      'index.html',
      'wasm_notebook_runtime.js',
      'compiler_worker_bootstrap.js',
      'editor_client.js',
      'editor_client.wasm',
      'editor_client.mjs',
    ]) {
      final src = File(p.join(webDir.path, fileName));
      if (src.existsSync()) {
        src.copySync(p.join(targetDir.path, fileName));
      }
    }
  }

  bool _stageDartPadAssetsIfAvailable(Directory targetDir) {
    final candidates = <String>[
      if (dartSdkRepoCheckout != null)
        p.join(dartSdkRepoCheckout!, 'out', 'ReleaseX64', 'dartpad'),
      '/usr/local/google/home/sigurdm/projects/dart-sdk/sdk/out/ReleaseX64/dartpad',
    ];
    for (final dirPath in candidates) {
      final dir = Directory(dirPath);
      final workerWasm = File(p.join(dirPath, 'worker.wasm'));
      final workerMjs = File(p.join(dirPath, 'worker.mjs'));
      final sdkTar = File(p.join(dirPath, 'dart', 'sdk.tar'));
      if (dir.existsSync() &&
          workerWasm.existsSync() &&
          workerMjs.existsSync() &&
          sdkTar.existsSync()) {
        final dstDartPad = Directory(p.join(targetDir.path, 'dartpad'));
        final dstDartSub = Directory(p.join(dstDartPad.path, 'dart'));
        dstDartSub.createSync(recursive: true);
        workerWasm.copySync(p.join(dstDartPad.path, 'worker.wasm'));
        workerMjs.copySync(p.join(dstDartPad.path, 'worker.mjs'));
        final loaderJs = File(p.join(dirPath, 'worker.loader.js'));
        if (loaderJs.existsSync()) {
          loaderJs.copySync(p.join(dstDartPad.path, 'worker.loader.js'));
        }
        sdkTar.copySync(p.join(dstDartSub.path, 'sdk.tar'));
        return true;
      }
    }
    return false;
  }

  String _generateCompilerWorkerDartSource() => r'''
// Generated by WasmNotebookBuilder; do not edit by hand.
import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:dart2wasm/compile.dart';
import 'package:dart2wasm/compiler_options.dart';
import 'package:dart2wasm/io_util.dart';
import 'package:front_end/src/api_prototype/file_system.dart';
import 'package:front_end/src/api_prototype/memory_file_system.dart';
import 'package:front_end/src/api_unstable/vm.dart'
    show CfeDiagnosticMessage, printDiagnosticMessage, enableColors;
import 'package:kernel/ast.dart' show Component;
import 'package:kernel/binary/ast_from_binary.dart'
    show BinaryBuilderWithMetadata;
import 'package:notebook/src/wasm_cell_bundler.dart';
import 'package:path/path.dart' as p;

@JS('dartCompilerInit')
external set _dartCompilerInit(JSFunction fn);

@JS('dartCompilerCompile')
external set _dartCompilerCompile(JSFunction fn);

@JS('dartCompilerBundleForAnalysis')
external set _dartCompilerBundleForAnalysis(JSFunction fn);

@JS('dartCompilerLastWasmBytes')
external set _dartCompilerLastWasmBytes(JSUint8Array? bytes);

@JS('dartCompilerLastMjsText')
external set _dartCompilerLastMjsText(JSString? text);

final class _CapturingIoManager extends CompilerPhaseInputOutputManager {
  final Map<String, Uint8List> outputs = {};

  _CapturingIoManager(super.fileSystem, super.options);

  Future<Uint8List> _read(Uri uri) =>
      fileSystem.entityForUri(uri).readAsBytes();

  @override
  Future<String> readString(Uri uri) async => utf8.decode(await _read(uri));

  @override
  Future<List<int>> readBytes(Uri uri) => _read(uri);

  @override
  Future<void> readComponent(Uri componentUri, Component component) async {
    BinaryBuilderWithMetadata(await _read(componentUri)).readComponent(component);
  }

  @override
  Future<void> writeComponent(
    Component component,
    String path, {
    bool includeSource = true,
  }) {
    throw UnsupportedError('writeComponent($path)');
  }

  @override
  void writeComponentAsText(Component component, String path) {
    throw UnsupportedError('writeComponentAsText($path)');
  }

  @override
  Future<void> writeWasmModule(Uint8List wasmModule, String moduleName) async {
    outputs[moduleName] = wasmModule;
  }

  @override
  Future<void> writeWasmSourceMap(String sourceMap, String moduleName) async {
    outputs['$moduleName.map'] = utf8.encode(sourceMap);
  }

  @override
  Future<void> writeJsRuntime(String jsRuntime) async {
    outputs[p.setExtension(p.basename(options.outputFile), '.mjs')] =
        utf8.encode(jsRuntime);
  }

  @override
  Future<void> writeSupportJs(String supportJs) async {
    outputs[p.setExtension(p.basename(options.outputFile), '.support.js')] =
        utf8.encode(supportJs);
  }

  @override
  Future<void> runWasmOpt(
    String mainWasmModule,
    int moduleId,
    List<String> flags,
  ) {
    throw StateError('runWasmOpt must not be reached in the embedded compiler');
  }

  @override
  Future<Set<int>> getModuleIds(String mainWasmFilePath) {
    throw UnsupportedError('getModuleIds');
  }

  @override
  Future<Uri?> resolveUri(Uri? uri) async => uri;
}

final class _CachedCompileArtifact {
  final Uint8List wasmBytes;
  final String mjsText;
  const _CachedCompileArtifact({
    required this.wasmBytes,
    required this.mjsText,
  });
}

MemoryFileSystem? _memFs;
final Map<String, _CachedCompileArtifact> _compileCache = {};

JSPromise<JSString> _handleInit(
  JSUint8Array platformDillBytes,
  JSString sourcesBundleJson,
) {
  return Future<JSString>(() async {
    final sw = Stopwatch()..start();
    enableColors = false;
    final fs = MemoryFileSystem(Uri.parse('memory:///work/'));
    fs
        .entityForUri(Uri.parse('memory:///sdk/dart2wasm_platform.dill'))
        .writeAsBytesSync(platformDillBytes.toDart);

    final bundle =
        jsonDecode(sourcesBundleJson.toDart) as Map<String, Object?>;
    final packagesJson = bundle['packagesJson'] as String;
    fs
        .entityForUri(Uri.parse('memory:///work/.dart_tool/package_config.json'))
        .writeAsStringSync(packagesJson);

    final files = (bundle['files'] as Map<String, Object?>);
    var fileCount = 0;
    for (final entry in files.entries) {
      final path = entry.key.startsWith('/') ? entry.key : '/${entry.key}';
      fs
          .entityForUri(Uri.parse('memory://$path'))
          .writeAsStringSync(entry.value as String);
      fileCount++;
    }
    _memFs = fs;
    sw.stop();
    return jsonEncode({
      'ok': true,
      'fileCount': fileCount,
      'initMs': sw.elapsedMilliseconds,
    }).toJS;
  }).toJS;
}

JSPromise<JSString> _handleCompile(JSString requestJson) {
  return Future<JSString>(() async {
    final fs = _memFs;
    if (fs == null) {
      return jsonEncode({
        'ok': false,
        'error': 'Compiler worker has not been initialized yet.',
      }).toJS;
    }

    final req = jsonDecode(requestJson.toDart) as Map<String, Object?>;
    String mainDartSource;
    List<String> includedCellIds = const [];
    List<String> declaredVariables = const [];

    if (req['cells'] case final List<Object?> rawCells) {
      final cells = rawCells
          .cast<Map<String, Object?>>()
          .map(WasmNotebookCell.fromJson)
          .toList();
      final targetCellId = req['targetCellId'] as String?;
      final bundled = WasmCellBundler.bundleCells(
        cells,
        targetCellId: targetCellId,
      );
      mainDartSource = bundled.mainDartSource;
      includedCellIds = bundled.includedCellIds;
      declaredVariables = bundled.declaredVariables;
    } else if (req['sourceCode'] case final String rawSource) {
      mainDartSource = rawSource;
    } else {
      return jsonEncode({
        'ok': false,
        'error': 'Compile request must include "cells" or "sourceCode".',
      }).toJS;
    }

    if (_compileCache[mainDartSource] case final cached?) {
      _dartCompilerLastWasmBytes = cached.wasmBytes.toJS;
      _dartCompilerLastMjsText = cached.mjsText.toJS;
      return jsonEncode({
        'ok': true,
        'cached': true,
        'compileMs': 0,
        'includedCellIds': includedCellIds,
        'declaredVariables': declaredVariables,
        'bundledSource': mainDartSource,
      }).toJS;
    }

    final sw = Stopwatch()..start();
    final mainUri = Uri.parse('memory:///work/cell_main.dart');
    fs.entityForUri(mainUri).writeAsStringSync(mainDartSource);

    final options = WasmCompilerOptions(
      mainUri: mainUri,
      outputFile: 'memory:///out/cell.wasm',
    )
      ..packagesPath = Uri.parse('memory:///work/.dart_tool/package_config.json')
      ..platformPath = Uri.parse('memory:///sdk/dart2wasm_platform.dill');
    options.translatorOptions
      ..optimizationLevel = 0
      ..enableExperimentalFfi = true
      ..generateSourceMaps = false;
    options.validate();

    final ioManager = _CapturingIoManager(fs, options);
    final diagnostics = <String>[];
    final result = await compile(options, ioManager, (CfeDiagnosticMessage m) {
      printDiagnosticMessage(m, diagnostics.add);
    });
    sw.stop();

    if (result is CompilationSuccess) {
      final wasmBytes = ioManager.outputs['cell.wasm'];
      final mjsBytes = ioManager.outputs['cell.mjs'];
      if (wasmBytes == null || mjsBytes == null) {
        return jsonEncode({
          'ok': false,
          'compileMs': sw.elapsedMilliseconds,
          'error': 'Compiler succeeded but produced no cell.wasm/cell.mjs.',
          'diagnostics': diagnostics,
        }).toJS;
      }
      final mjsText = utf8.decode(mjsBytes);
      if (_compileCache.length >= 16) {
        _compileCache.remove(_compileCache.keys.first);
      }
      _compileCache[mainDartSource] = _CachedCompileArtifact(
        wasmBytes: wasmBytes,
        mjsText: mjsText,
      );
      _dartCompilerLastWasmBytes = wasmBytes.toJS;
      _dartCompilerLastMjsText = mjsText.toJS;
      return jsonEncode({
        'ok': true,
        'cached': false,
        'compileMs': sw.elapsedMilliseconds,
        'includedCellIds': includedCellIds,
        'declaredVariables': declaredVariables,
        'bundledSource': mainDartSource,
      }).toJS;
    }

    var errorMsg = diagnostics.join('\n').trim();
    if (errorMsg.isEmpty) {
      if (result is CFECrashError) {
        errorMsg = 'CFE Crash: ${result.error}\n${result.stackTrace}';
      } else {
        errorMsg = 'Compilation failed (${result.runtimeType}).';
      }
    }
    _dartCompilerLastWasmBytes = null;
    _dartCompilerLastMjsText = null;
    return jsonEncode({
      'ok': false,
      'compileMs': sw.elapsedMilliseconds,
      'error': errorMsg,
      'diagnostics': diagnostics,
      'bundledSource': mainDartSource,
    }).toJS;
  }).toJS;
}

JSString _handleBundleForAnalysis(JSString requestJson) {
  final req = jsonDecode(requestJson.toDart) as Map<String, Object?>;
  final rawCells = (req['cells'] as List<Object?>).cast<Map<String, Object?>>();
  final cells = rawCells.map(WasmNotebookCell.fromJson).toList();
  final activeCellId = (req['activeCellId'] as String?) ?? '';
  final cursorOffset = (req['cursorOffset'] as num?)?.toInt() ?? 0;
  final bundle = WasmCellBundler.bundleForAnalysis(
    cells,
    activeCellId: activeCellId,
    cursorOffsetInCell: cursorOffset,
  );
  return jsonEncode({
    'source': bundle.source,
    'mappedOffset': bundle.mappedOffset,
  }).toJS;
}

void main() {
  enableColors = false;
  _dartCompilerInit = _handleInit.toJS;
  _dartCompilerCompile = _handleCompile.toJS;
  _dartCompilerBundleForAnalysis = _handleBundleForAnalysis.toJS;
}
''';
}
