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

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:vm_service/vm_service.dart';
import 'package:vm_service/vm_service_io.dart';
import 'package:notebook/src/cell_formatter.dart';
import 'package:notebook/src/lsp_client.dart';
import 'package:notebook/src/kernel_helper.dart';
import 'package:notebook/src/wasm_cell_bundler.dart';

class CompletionItem {
  final String label;
  final String type;
  final String? detail;

  CompletionItem({required this.label, required this.type, this.detail});

  Map<String, dynamic> toJson() => {
    'label': label,
    'type': type,
    if (detail != null) 'detail': detail,
  };
}

class NotebookKernel {
  final String workspaceDir;
  final String dartSdkPath;

  Process? _process;
  VmService? _service;
  String? _isolateId;
  String? _workspaceLibId;

  LspClient? _lspClient;
  int _workspaceVersion = 1;
  File? _savedNativeAssetsFile;
  String? _savedNativeAssetsContent;
  String? _savedWorkspaceContent;

  static const String _gpuArrayDefaultImport =
      "import 'package:gpuarray/gpuarray.dart' show "
      'BrowserWebGpuBackend, GradFn, GpuArray, GpuArrayBaseDivide, '
      'GpuArrayBFloat16ReductionExtension, GpuArrayBitwise, '
      'GpuArrayBitwiseSpec, GpuArrayComplex128ReductionExtension, '
      'GpuArrayComplex64ReductionExtension, GpuArrayDefaultComponentExtension, '
      'GpuArrayDefaultReductionExtension, GpuArrayDivide, '
      'GpuArrayFloat16ReductionExtension, GpuArrayFloat32ReductionExtension, '
      'GpuArrayFloat64ReductionExtension, GpuArrayNDArrayInterop, '
      'GpuArrayShift, GpuArrayShiftSpec, GpuArraySpecComponentExtension, '
      'GpuArrayTypedOperationsExtension, GpuBackend, GpuBuffer, '
      'GpuBufferUsage, GpuDevice, GpuDeviceDisposedException, '
      'GpuDeviceException, GpuDeviceType, GpuException, GpuMemoryException, '
      'GpuMemoryPool, GpuShaderCompilationException, '
      'GpuShapeMismatchException, GpuSlice, LossReduction, NDArrayGpuInterop, '
      'createDefaultGpuBackend, createWebGpuDevice, enableGrad, isGradEnabled, '
      'noGrad;';

  static const String _gpuArrayJitDefaultImport =
      "import 'package:gpuarray/jit.dart' hide Expr;";

  final Set<String> _imports = {
    "import 'package:ndarray/ndarray.dart';",
    _gpuArrayDefaultImport,
    _gpuArrayJitDefaultImport,
    "import 'dart:math' as math;",
  };
  final Map<String, String> _definitions = {};
  final Set<String> _declaredVariables = {};

  /// Chain of pending [execute] calls; cells never run concurrently.
  Future<void> _executionQueue = Future<void>.value();

  /// Identifier handed to `runNotebookCell` for the most recently started cell.
  int _cellRunCounter = 0;

  /// Completion signal for the in-flight cell run, if any.
  Completer<void>? _cellDone;
  int? _inFlightRunId;

  NotebookKernel({required this.workspaceDir, required this.dartSdkPath}) {
    final wf = _getWorkspaceFile();
    if (wf.existsSync()) {
      _savedWorkspaceContent = wf.readAsStringSync();
    }
    _writeWorkspace();
  }

  File? _findNativeAssetsYaml() {
    var dir = Directory(workspaceDir).absolute;
    for (var i = 0; i < 4; i++) {
      final candidate = File(
        p.join(dir.path, '.dart_tool', 'native_assets.yaml'),
      );
      if (candidate.existsSync()) return candidate;
      final parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
    return null;
  }

  static bool _nativeAssetsPathsExist(String yamlContent) {
    final matches = RegExp(
      r'"(/tmp/dart_native_assets_[^"]+)"',
    ).allMatches(yamlContent);
    if (matches.isEmpty) return true;
    for (final m in matches) {
      if (!File(m.group(1)!).existsSync()) return false;
    }
    return true;
  }

  static String _stabilizeNativeAssetsContent(
    File yamlFile,
    String yamlContent,
  ) {
    final cacheDir = Directory(
      p.join(yamlFile.parent.path, 'notebook_native_assets'),
    );
    var updated = yamlContent;
    final matches = RegExp(
      r'"(/tmp/dart_native_assets_[^"]+)"',
    ).allMatches(yamlContent).toList();
    for (final m in matches) {
      final srcPath = m.group(1)!;
      final srcFile = File(srcPath);
      if (!srcFile.existsSync()) continue;
      try {
        if (!cacheDir.existsSync()) {
          cacheDir.createSync(recursive: true);
        }
        final dstPath = p.join(cacheDir.path, p.basename(srcPath));
        final dstFile = File(dstPath);
        if (!dstFile.existsSync() ||
            dstFile.lengthSync() != srcFile.lengthSync()) {
          final tmpFile = File('$dstPath.tmp.$pid');
          srcFile.copySync(tmpFile.path);
          tmpFile.renameSync(dstPath);
        }
        updated = updated.replaceAll('"$srcPath"', '"$dstPath"');
      } catch (_) {}
    }
    return updated;
  }

  void _saveNativeAssetsYaml() {
    final file = _findNativeAssetsYaml();
    if (file != null && file.existsSync()) {
      try {
        final raw = file.readAsStringSync();
        if (_nativeAssetsPathsExist(raw)) {
          final stabilized = _stabilizeNativeAssetsContent(file, raw);
          if (stabilized != raw) {
            file.writeAsStringSync(stabilized);
          }
          _savedNativeAssetsFile = file;
          _savedNativeAssetsContent = stabilized;
        }
      } catch (_) {}
    }
  }

  void _restoreNativeAssetsYaml() {
    final file = _savedNativeAssetsFile ?? _findNativeAssetsYaml();
    if (file != null && file.existsSync()) {
      try {
        final current = file.readAsStringSync();
        if (_nativeAssetsPathsExist(current)) {
          final stabilized = _stabilizeNativeAssetsContent(file, current);
          if (stabilized != current) {
            file.writeAsStringSync(stabilized);
          }
          _savedNativeAssetsFile = file;
          _savedNativeAssetsContent = stabilized;
          return;
        }
      } catch (_) {}
    }
    final savedFile = _savedNativeAssetsFile;
    final content = _savedNativeAssetsContent;
    if (savedFile != null &&
        content != null &&
        _nativeAssetsPathsExist(content)) {
      try {
        savedFile.writeAsStringSync(content);
      } catch (_) {}
    }
  }

  File _getWorkspaceFile() {
    final candidate2 = File(
      p.join(workspaceDir, 'pkgs', 'notebook', 'lib', 'src', 'workspace.dart'),
    );
    if (candidate2.existsSync() ||
        File(
          p.join(workspaceDir, 'pkgs', 'notebook', 'pubspec.yaml'),
        ).existsSync()) {
      return candidate2;
    }
    return File(p.join(workspaceDir, 'lib', 'src', 'workspace.dart'));
  }

  Future<void> start() async {
    await _startKernelOnly();

    _lspClient = LspClient(dartSdkPath: dartSdkPath, rootPath: workspaceDir);
    try {
      await _lspClient!.start();
      final workspaceFile = _getWorkspaceFile();
      final fileUri = p.toUri(workspaceFile.path).toString();
      _lspClient!.didOpen(
        fileUri,
        workspaceFile.existsSync() ? workspaceFile.readAsStringSync() : '',
      );
    } catch (e) {
      print('Warning: Failed to start LSP client: $e');
    }
  }

  Future<void> _startKernelOnly() async {
    _saveNativeAssetsYaml();
    final dartExecutable = p.join(dartSdkPath, 'bin', 'dart');
    var kernelScriptPath = p.join(workspaceDir, 'bin', 'kernel.dart');
    if (!File(kernelScriptPath).existsSync()) {
      kernelScriptPath = p.join(
        workspaceDir,
        'pkgs',
        'notebook',
        'bin',
        'kernel.dart',
      );
    }

    _process = await Process.start(dartExecutable, [
      '--enable-vm-service=0',
      kernelScriptPath,
    ], workingDirectory: workspaceDir);

    final uriCompleter = Completer<String>();

    _process!.stdout
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen((line) {
          final match = RegExp(
            r'The Dart VM service is listening on (http://\S+/)',
          ).firstMatch(line);
          if (match != null && !uriCompleter.isCompleted) {
            uriCompleter.complete(match.group(1)!);
          }
        });

    _process!.stderr
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen((line) {
          final match = RegExp(
            r'The Dart VM service is listening on (http://\S+/)',
          ).firstMatch(line);
          if (match != null && !uriCompleter.isCompleted) {
            uriCompleter.complete(match.group(1)!);
          }
        });

    final vmServiceUri = await uriCompleter.future.timeout(
      const Duration(seconds: 10),
      onTimeout: () => throw TimeoutException(
        'Failed to find VM Service URI from kernel process',
      ),
    );
    _restoreNativeAssetsYaml();

    final wsUri = '${vmServiceUri.replaceFirst('http://', 'ws://')}ws';
    _service = await vmServiceConnectUri(wsUri);

    var vm = await _service!.getVM();
    IsolateRef? mainIsolateRef;
    final timeout = DateTime.now().add(const Duration(seconds: 30));

    while (mainIsolateRef == null) {
      if (DateTime.now().isAfter(timeout)) {
        throw TimeoutException(
          'Timed out waiting for main isolate in kernel process.',
        );
      }
      vm = await _service!.getVM();
      for (final isolateRef in vm.isolates ?? <IsolateRef>[]) {
        if (isolateRef.name == 'main' || vm.isolates!.length == 1) {
          mainIsolateRef = isolateRef;
          break;
        }
      }
      if (mainIsolateRef == null) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
      }
    }

    _isolateId = mainIsolateRef.id;

    final isolateTimeout = DateTime.now().add(const Duration(seconds: 30));
    var isolate = await _service!.getIsolate(_isolateId!);
    while (isolate.runnable != true || isolate.rootLib == null) {
      if (DateTime.now().isAfter(isolateTimeout)) {
        throw TimeoutException(
          'Timed out waiting for runnable isolate rootLib.',
        );
      }
      await Future<void>.delayed(const Duration(milliseconds: 100));
      isolate = await _service!.getIsolate(_isolateId!);
    }

    _updateWorkspaceLibId(isolate);

    final service = _service!;
    await service.streamListen(EventStreams.kExtension);
    service.onExtensionEvent.listen(_onExtensionEvent);
    unawaited(
      service.onDone.then((_) {
        _failInFlightCell(StateError('Kernel VM service connection closed'));
      }),
    );
  }

  void _onExtensionEvent(Event event) {
    if (event.extensionKind != notebookCellDoneEvent) return;
    if (event.extensionData?.data['runId'] != _inFlightRunId) return;
    final done = _cellDone;
    if (done != null && !done.isCompleted) done.complete();
  }

  void _failInFlightCell(Object error) {
    final done = _cellDone;
    if (done != null && !done.isCompleted) done.completeError(error);
  }

  Future<void> _restartKernelProcess() async {
    await _service?.dispose();
    final proc = _process;
    if (proc != null) {
      proc.kill();
      try {
        await proc.exitCode.timeout(const Duration(seconds: 2));
      } catch (_) {}
    }
    _restoreNativeAssetsYaml();
    _process = null;
    _service = null;

    await _startKernelOnly();
    await _reloadWorkspace();
  }

  Future<String> _handleAddDependency(String pkgName) async {
    var targetWorkingDir = workspaceDir;
    if (File(
      p.join(workspaceDir, 'pkgs', 'notebook', 'pubspec.yaml'),
    ).existsSync()) {
      targetWorkingDir = p.join(workspaceDir, 'pkgs', 'notebook');
    }
    final dartExecutable = p.join(dartSdkPath, 'bin', 'dart');
    final result = await Process.run(dartExecutable, [
      'pub',
      'add',
      pkgName,
    ], workingDirectory: targetWorkingDir);

    if (result.exitCode != 0) {
      return 'Failed to add dependency "$pkgName":\n${result.stderr}';
    }

    try {
      await _restartKernelProcess();
      return 'Successfully added package "$pkgName" and reloaded environment.';
    } catch (e) {
      return 'Added package "$pkgName", but failed to reload kernel: $e';
    }
  }

  Future<void> _ensurePackageInstalled(String pkgName) async {
    var pubspecFile = File(p.join(workspaceDir, 'pubspec.yaml'));
    if (!pubspecFile.existsSync() &&
        File(
          p.join(workspaceDir, 'pkgs', 'notebook', 'pubspec.yaml'),
        ).existsSync()) {
      pubspecFile = File(
        p.join(workspaceDir, 'pkgs', 'notebook', 'pubspec.yaml'),
      );
    }
    if (pubspecFile.existsSync()) {
      final content = pubspecFile.readAsStringSync();
      if (content.contains('$pkgName:')) return;
    }
    await _handleAddDependency(pkgName);
  }

  void _updateWorkspaceLibId(Isolate isolate) {
    LibraryRef? targetLib;
    for (final lib in isolate.libraries ?? <LibraryRef>[]) {
      if (lib.uri != null && lib.uri!.endsWith('workspace.dart')) {
        targetLib = lib;
      }
    }
    if (targetLib != null) {
      _workspaceLibId = targetLib.id;
      return;
    }
    throw StateError('Could not find workspace.dart library in isolate');
  }

  Future<void> stop() async {
    await _lspClient?.stop();
    await _service?.dispose();
    final proc = _process;
    if (proc != null) {
      proc.kill();
      try {
        await proc.exitCode.timeout(const Duration(seconds: 2));
      } catch (_) {}
    }
    _restoreNativeAssetsYaml();
    if (_savedWorkspaceContent != null) {
      try {
        _getWorkspaceFile().writeAsStringSync(_savedWorkspaceContent!);
      } catch (_) {}
    }
  }

  /// Formats Dart [code] using [formatNotebookCellCode].
  String formatCode(String code) => formatNotebookCellCode(code);

  /// Executes the notebook cell [code] and returns its outputs.
  ///
  /// For an ordinary cell the result is a JSON list of `CellOutputItem`
  /// objects: captured `print`/`display` output followed by the formatted
  /// value of the trailing expression, or an `Error:` item if the cell threw.
  /// Declarations, imports and `pub add` commands return a plain status line.
  ///
  /// Cells run inside an `async` closure in the kernel isolate, so `await` may
  /// be used anywhere in a cell and a trailing `Future` is awaited before its
  /// value is shown. Calls are serialized: a cell starts only after the
  /// previous one, including its asynchronous work, has completed.
  ///
  /// Throws if the cell fails to compile or the kernel connection is lost.
  Future<String> execute(String code) {
    final previous = _executionQueue;
    final completer = Completer<void>();
    _executionQueue = completer.future;
    return previous
        .then((_) => _executeUnqueued(code))
        .whenComplete(completer.complete);
  }

  Future<String> _executeUnqueued(String code) async {
    final parsed = ParsedNotebookCell.parse(code);
    if (parsed.isEmpty) return '';

    if (parsed.pubAddPackage case final pkgName?) {
      return _handleAddDependency(pkgName);
    }

    if (parsed.imports.isNotEmpty) {
      for (final imp in parsed.imports) {
        if (imp.statement.contains('package:') &&
            imp.name != 'ndarray' &&
            imp.name != 'notebook' &&
            imp.name != 'symbolic_dart' &&
            imp.name != 'gpuarray' &&
            imp.name != 'resource_scope') {
          await _ensurePackageInstalled(imp.name);
        }
        _imports.add(imp.statement);
      }
      if (parsed.isPureImport) {
        await _reloadWorkspace();
        return parsed.importedNamesSummary;
      }
    }

    final prevDefs = Map<String, String>.of(_definitions);
    final prevVars = Set<String>.of(_declaredVariables);
    final transformRes = parsed.transformForKernel();
    _definitions.addAll(transformRes.namedDefinitions);
    _declaredVariables.addAll(transformRes.declaredVariables);
    try {
      await _reloadWorkspace();
    } catch (e) {
      _definitions
        ..clear()
        ..addAll(prevDefs);
      _declaredVariables
        ..clear()
        ..addAll(prevVars);
      _writeWorkspace();
      try {
        await _service!.reloadSources(_isolateId!);
      } catch (_) {}
      rethrow;
    }

    if (parsed.onlyDeclarations) {
      final sym = parsed.lastDeclaredSymbol;
      return sym != null ? 'Declared: $sym' : '';
    }

    final runId = ++_cellRunCounter;
    final done = Completer<void>();
    _inFlightRunId = runId;
    _cellDone = done;
    try {
      final evalRef = await _service!.evaluate(
        _isolateId!,
        _workspaceLibId!,
        'runNotebookCell($runId, () async {\n${transformRes.cellBodyCode}\n})',
      );
      if (evalRef is ErrorRef) {
        throw StateError('Evaluation failed: ${evalRef.message}');
      }
      await _waitForCellCompletion(done);

      final resultJson = await _evaluateString('notebookCellResultJson()');
      if (resultJson == null) {
        throw StateError('Kernel did not report a result for the cell.');
      }
      final result = jsonDecode(resultJson) as Map<String, dynamic>;
      final outputs = [
        for (final item in result['outputs'] as List<dynamic>)
          CellOutputItem.fromJson(
            Map<String, dynamic>.from(item as Map<dynamic, dynamic>),
          ),
      ];
      return jsonEncode(outputs.map((e) => e.toJson()).toList());
    } finally {
      _inFlightRunId = null;
      _cellDone = null;
    }
  }

  /// Waits until the in-flight cell has completed.
  ///
  /// Normally [done] is completed by the `notebook.cellDone` extension event.
  /// As a safety net against a lost event, the isolate is also asked directly
  /// every couple of seconds; the request is only answered once the isolate
  /// yields, so it does not interfere with a long-running synchronous cell.
  Future<void> _waitForCellCompletion(Completer<void> done) async {
    while (!done.isCompleted) {
      await done.future.timeout(const Duration(seconds: 2), onTimeout: () {});
      if (done.isCompleted) return;
      final isDone = await _service!.evaluate(
        _isolateId!,
        _workspaceLibId!,
        'notebookCellIsDone()',
      );
      if (isDone is InstanceRef && isDone.valueAsString == 'true') return;
    }
  }

  /// Evaluates [expression] in the workspace library and returns its `String`
  /// value in full, or `null` if the result is not a string.
  Future<String?> _evaluateString(String expression) async {
    final ref = await _service!.evaluate(
      _isolateId!,
      _workspaceLibId!,
      expression,
    );
    if (ref is! InstanceRef || ref.kind != InstanceKind.kString) return null;
    if (ref.valueAsStringIsTruncated == true && ref.id != null) {
      final full = await _service!.getObject(_isolateId!, ref.id!);
      if (full is Instance && full.valueAsString != null) {
        return full.valueAsString;
      }
    }
    return ref.valueAsString;
  }

  Future<List<CompletionItem>> getCompletions(
    String code,
    int cursorOffset,
  ) async {
    if (_lspClient != null) {
      try {
        final workspaceFile = _getWorkspaceFile();
        final fileUri = p.toUri(workspaceFile.path).toString();
        final baseContent = workspaceFile.existsSync()
            ? workspaceFile.readAsStringSync()
            : '';

        final prefix =
            '$baseContent\n\nFuture<void> __eval_dummy__() async {\n';
        final fullContent = '$prefix$code\n}';

        _lspClient!.didChange(fileUri, fullContent, ++_workspaceVersion);

        final targetOffset =
            prefix.length +
            (cursorOffset <= code.length ? cursorOffset : code.length);
        final lines = fullContent.substring(0, targetOffset).split('\n');
        final line = lines.length - 1;
        final character = lines.last.length;

        final lspResults = await _lspClient!.getCompletions(
          fileUri,
          line,
          character,
        );

        _lspClient!.didChange(fileUri, baseContent, ++_workspaceVersion);

        if (lspResults.isNotEmpty) {
          return lspResults
              .map(
                (r) => CompletionItem(
                  label: r.label,
                  type: r.kind,
                  detail: r.detail,
                ),
              )
              .toList();
        }
      } catch (_) {
        // Fall back
      }
    }

    return _getLocalCompletions(code, cursorOffset);
  }

  Future<String?> getHover(String code, int cursorOffset) async {
    if (_lspClient == null) return null;
    try {
      final workspaceFile = _getWorkspaceFile();
      final fileUri = p.toUri(workspaceFile.path).toString();
      final baseContent = workspaceFile.existsSync()
          ? workspaceFile.readAsStringSync()
          : '';

      final prefix = '$baseContent\n\nFuture<void> __eval_dummy__() async {\n';
      final fullContent = '$prefix$code\n}';

      _lspClient!.didChange(fileUri, fullContent, ++_workspaceVersion);

      final targetOffset =
          prefix.length +
          (cursorOffset <= code.length ? cursorOffset : code.length);
      final lines = fullContent.substring(0, targetOffset).split('\n');
      final line = lines.length - 1;
      final character = lines.last.length;

      final hoverResult = await _lspClient!.getHover(fileUri, line, character);

      _lspClient!.didChange(fileUri, baseContent, ++_workspaceVersion);

      return hoverResult;
    } catch (_) {
      return null;
    }
  }

  List<CompletionItem> _getLocalCompletions(String code, int cursorOffset) {
    if (cursorOffset > code.length) cursorOffset = code.length;
    final textBeforeCursor = code.substring(0, cursorOffset);

    // Check if user is typing after a dot e.g. "a." or "NDArray."
    final dotMatch = RegExp(
      r'([\w\d_$]+)\.([\w\d_$]*)$',
    ).firstMatch(textBeforeCursor);
    if (dotMatch != null) {
      final target = dotMatch.group(1)!;
      final prefix = dotMatch.group(2)!;

      final items = <CompletionItem>[];
      if (target == 'NDArray') {
        items.addAll([
          CompletionItem(
            label: 'fromList',
            type: 'constructor',
            detail:
                'factory NDArray.fromList(List list, List<int> shape, DType dtype)',
          ),
          CompletionItem(
            label: 'zeros',
            type: 'constructor',
            detail: 'factory NDArray.zeros(List<int> shape, DType dtype)',
          ),
          CompletionItem(
            label: 'ones',
            type: 'constructor',
            detail: 'factory NDArray.ones(List<int> shape, DType dtype)',
          ),
          CompletionItem(
            label: 'arange',
            type: 'constructor',
            detail:
                'factory NDArray.arange(num start, num stop, [num step, DType dtype])',
          ),
          CompletionItem(
            label: 'linspace',
            type: 'constructor',
            detail:
                'factory NDArray.linspace(num start, num stop, int num, [DType dtype])',
          ),
          CompletionItem(
            label: 'scope',
            type: 'method',
            detail: 'static T scope<T>(T Function() fn)',
          ),
        ]);
      } else if (target == 'DType') {
        items.addAll([
          CompletionItem(
            label: 'float64',
            type: 'property',
            detail: 'DType<Float64>',
          ),
          CompletionItem(
            label: 'float32',
            type: 'property',
            detail: 'DType<Float32>',
          ),
          CompletionItem(
            label: 'int64',
            type: 'property',
            detail: 'DType<Int64>',
          ),
          CompletionItem(
            label: 'int32',
            type: 'property',
            detail: 'DType<Int32>',
          ),
          CompletionItem(
            label: 'int16',
            type: 'property',
            detail: 'DType<Int16>',
          ),
          CompletionItem(
            label: 'uint8',
            type: 'property',
            detail: 'DType<Uint8>',
          ),
          CompletionItem(
            label: 'complex128',
            type: 'property',
            detail: 'DType<Complex128>',
          ),
          CompletionItem(
            label: 'complex64',
            type: 'property',
            detail: 'DType<Complex64>',
          ),
          CompletionItem(
            label: 'boolean',
            type: 'property',
            detail: 'DType<bool>',
          ),
        ]);
      } else {
        items.addAll([
          CompletionItem(
            label: 'shape',
            type: 'property',
            detail: 'List<int> shape',
          ),
          CompletionItem(
            label: 'dtype',
            type: 'property',
            detail: 'DType<T> dtype',
          ),
          CompletionItem(label: 'size', type: 'property', detail: 'int size'),
          CompletionItem(label: 'rank', type: 'property', detail: 'int rank'),
          CompletionItem(
            label: 'isContiguous',
            type: 'property',
            detail: 'bool isContiguous',
          ),
          CompletionItem(
            label: 'isSquare',
            type: 'property',
            detail: 'bool isSquare',
          ),
          CompletionItem(label: 'scalar', type: 'property', detail: 'T scalar'),
          CompletionItem(
            label: 'transposed',
            type: 'property',
            detail: 'NDArray<T> transposed',
          ),
          CompletionItem(
            label: 'copy()',
            type: 'method',
            detail: 'NDArray<T> copy()',
          ),
          CompletionItem(
            label: 'slice()',
            type: 'method',
            detail: 'NDArray<T> slice(List selectors)',
          ),
          CompletionItem(
            label: 'reshape()',
            type: 'method',
            detail: 'NDArray<T> reshape(List<int> newShape)',
          ),
          CompletionItem(
            label: 'transpose()',
            type: 'method',
            detail: 'NDArray<T> transpose([List<int> axes])',
          ),
          CompletionItem(
            label: 'flatten()',
            type: 'method',
            detail: 'NDArray<T> flatten()',
          ),
          CompletionItem(
            label: 'ravel()',
            type: 'method',
            detail: 'NDArray<T> ravel()',
          ),
          CompletionItem(
            label: 'dispose()',
            type: 'method',
            detail: 'void dispose()',
          ),
          CompletionItem(
            label: 'diff(Symbol x)',
            type: 'method',
            detail: 'Expr diff(Symbol x)',
          ),
          CompletionItem(
            label: 'expand()',
            type: 'method',
            detail: 'Expr expand()',
          ),
          CompletionItem(
            label: 'subs(Map map)',
            type: 'method',
            detail: 'Expr subs(Map<Object, Object> map)',
          ),
          CompletionItem(
            label: 'lambdify(List<Symbol> vars)',
            type: 'method',
            detail: 'SymbolicLambda lambdify(List<Symbol> variables)',
          ),
          CompletionItem(
            label: 'toLatex()',
            type: 'method',
            detail: 'String toLatex()',
          ),
          CompletionItem(
            label: 'toCCode()',
            type: 'method',
            detail: 'String toCCode()',
          ),
          CompletionItem(label: 'det()', type: 'method', detail: 'Expr det()'),
          CompletionItem(
            label: 'inv()',
            type: 'method',
            detail: 'SymbolicMatrix inv()',
          ),
          CompletionItem(
            label: 'solve(SymbolicMatrix b)',
            type: 'method',
            detail: 'SymbolicMatrix solve(SymbolicMatrix b)',
          ),
          CompletionItem(
            label: 'factor()',
            type: 'method',
            detail: 'PolyFactorization factor()',
          ),
        ]);
      }
      return items
          .where((i) => i.label.toLowerCase().startsWith(prefix.toLowerCase()))
          .toList();
    }

    final wordMatch = RegExp(r'([\w\d_$]+)$').firstMatch(textBeforeCursor);
    final prefix = wordMatch != null ? wordMatch.group(1)! : '';

    final items = <CompletionItem>[];
    for (final symbol in _definitions.keys) {
      final isVar = _declaredVariables.contains(symbol);
      items.add(
        CompletionItem(
          label: symbol,
          type: isVar ? 'variable' : 'function',
          detail: 'User defined in workspace',
        ),
      );
    }

    items.addAll([
      CompletionItem(
        label: 'Symbol',
        type: 'class',
        detail: 'Symbolic variable: Symbol(String name)',
      ),
      CompletionItem(
        label: 'Integer',
        type: 'class',
        detail: 'Exact integer symbolic node: Integer(int value)',
      ),
      CompletionItem(
        label: 'Real',
        type: 'class',
        detail: 'Real floating-point symbolic node: Real(double value)',
      ),
      CompletionItem(
        label: 'Rational',
        type: 'class',
        detail: 'Exact rational number: Rational(int num, int den)',
      ),
      CompletionItem(
        label: 'Expr',
        type: 'class',
        detail: 'Symbolic expression CAS node',
      ),
      CompletionItem(
        label: 'SymbolicMatrix',
        type: 'class',
        detail: 'Dense symbolic matrix: CDenseMatrix',
      ),
      CompletionItem(
        label: 'FlintRationalPoly',
        type: 'class',
        detail: 'Exact polynomial over Q[x] with factorization',
      ),
      CompletionItem(
        label: 'SymbolicOptimizer',
        type: 'class',
        detail: 'Newton-Raphson & Gradient Descent auto-diff solvers',
      ),
      CompletionItem(
        label: 'diff',
        type: 'function',
        detail: 'Expr diff(Expr f, Symbol x)',
      ),
      CompletionItem(
        label: 'expand',
        type: 'function',
        detail: 'Expr expand(Expr f)',
      ),
      CompletionItem(
        label: 'subs',
        type: 'function',
        detail: 'Expr subs(Expr f, Map<Object, Object> substitutions)',
      ),
      CompletionItem(
        label: 'lambdify',
        type: 'function',
        detail: 'SymbolicLambda lambdify(Expr f, List<Symbol> vars)',
      ),
      CompletionItem(
        label: 'evaluateSymbolic',
        type: 'function',
        detail: 'NDArray evaluateSymbolic(Expr f, Map<Symbol, NDArray> inputs)',
      ),
      CompletionItem(
        label: 'plotSymbolic',
        type: 'function',
        detail: 'Plot plotSymbolic(Expr f, Symbol var, {from, to, points})',
      ),
      CompletionItem(
        label: 'plotSymbolic2D',
        type: 'function',
        detail: 'Heatmap plotSymbolic2D(Expr f, Symbol x, Symbol y, ...)',
      ),
      CompletionItem(
        label: 'sum',
        type: 'function',
        detail: 'NDArray sum(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'mean',
        type: 'function',
        detail: 'NDArray mean(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'min',
        type: 'function',
        detail: 'NDArray min(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'max',
        type: 'function',
        detail: 'NDArray max(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'std',
        type: 'function',
        detail: 'NDArray std(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'dot',
        type: 'function',
        detail: 'NDArray dot(NDArray a, NDArray b)',
      ),
      CompletionItem(
        label: 'matmul',
        type: 'function',
        detail: 'NDArray matmul(NDArray a, NDArray b)',
      ),
      CompletionItem(
        label: 'sin',
        type: 'function',
        detail: 'NDArray sin(NDArray a) / Expr sin(Expr a)',
      ),
      CompletionItem(
        label: 'cos',
        type: 'function',
        detail: 'NDArray cos(NDArray a) / Expr cos(Expr a)',
      ),
      CompletionItem(
        label: 'exp',
        type: 'function',
        detail: 'NDArray exp(NDArray a) / Expr exp(Expr a)',
      ),
      CompletionItem(
        label: 'log',
        type: 'function',
        detail: 'NDArray log(NDArray a) / Expr log(Expr a)',
      ),
      CompletionItem(
        label: 'abs',
        type: 'function',
        detail: 'NDArray abs(NDArray a) / Expr abs(Expr a)',
      ),
      CompletionItem(
        label: 'sqrt',
        type: 'function',
        detail: 'NDArray sqrt(NDArray a) / Expr sqrt(Expr a)',
      ),
      CompletionItem(
        label: 'NDArray',
        type: 'class',
        detail: 'Multi-dimensional array',
      ),
      CompletionItem(
        label: 'Plot',
        type: 'class',
        detail: '2D line plot widget: Plot(x: xArr, y: yArr)',
      ),
      CompletionItem(
        label: 'Heatmap',
        type: 'class',
        detail: '2D heatmap widget: Heatmap(matrix2D)',
      ),
      CompletionItem(
        label: 'LaTeX',
        type: 'class',
        detail: 'Mathematical KaTeX equation display widget',
      ),
      CompletionItem(
        label: 'Image',
        type: 'class',
        detail: 'Renderable graphic image wrapper for NDArray',
      ),
      CompletionItem(
        label: 'GpuDevice',
        type: 'class',
        detail: 'Hardware or CPU compute device',
      ),
      CompletionItem(
        label: 'GpuArray',
        type: 'class',
        detail: 'GPU-accelerated n-dimensional array',
      ),
      CompletionItem(
        label: 'display',
        type: 'function',
        detail: 'Display an interactive widget, image, or plot',
      ),
      CompletionItem(
        label: 'DType',
        type: 'enum',
        detail: 'Data type specifier',
      ),
      CompletionItem(label: 'var', type: 'keyword'),
      CompletionItem(label: 'final', type: 'keyword'),
      CompletionItem(label: 'const', type: 'keyword'),
      CompletionItem(label: 'int', type: 'type'),
      CompletionItem(label: 'double', type: 'type'),
      CompletionItem(label: 'bool', type: 'type'),
      CompletionItem(label: 'String', type: 'type'),
      CompletionItem(label: 'List', type: 'type'),
    ]);

    return items
        .where((i) => i.label.toLowerCase().startsWith(prefix.toLowerCase()))
        .toList();
  }

  Future<void> _reloadWorkspace() async {
    _writeWorkspace();
    final reloadReport = await _service!.reloadSources(_isolateId!);
    if (reloadReport.success != true) {
      throw StateError(
        'Failed to reload workspace: ${reloadReport.json ?? reloadReport.toJson()}',
      );
    }
    final isolate = await _service!.getIsolate(_isolateId!);
    _updateWorkspaceLibId(isolate);
  }

  Future<List<Map<String, dynamic>>> getVariableInspectorData() async {
    if (_isolateId == null || _workspaceLibId == null || _service == null) {
      return [];
    }
    final results = <Map<String, dynamic>>[];
    for (final sym in _definitions.keys) {
      try {
        final evalObj = await _service!.evaluate(
          _isolateId!,
          _workspaceLibId!,
          '''
(() {
  try {
    final val = $sym;
    if (val is NDArray) {
      return 'NDArray shape: \${val.shape} | dtype: \${val.dtype.name} | strided: \${!val.isContiguous}';
    }
    if (val is Expr) {
      final str = '\$val';
      return 'Expr -> \${str.length > 60 ? str.substring(0, 60) + '...' : str}';
    }
    if (val is SymbolicMatrix) {
      return 'SymbolicMatrix [\${val.rows}x\${val.cols}]';
    }
    if (val is FlintRationalPoly) {
      final str = '\$val';
      return 'FlintRationalPoly degree: \${val.degree} -> \${str.length > 50 ? str.substring(0, 50) + '...' : str}';
    }
    final str = '\$val';
    return '\${val.runtimeType} -> \${str.length > 60 ? str.substring(0, 60) + '...' : str}';
  } catch (e) {
    return 'error: \$e';
  }
})()
''',
        );
        String? desc;
        if (evalObj is InstanceRef && evalObj.valueAsString != null) {
          desc = evalObj.valueAsString!;
        } else {
          desc = evalObj.toString();
        }
        results.add({
          'name': sym,
          'type': desc.startsWith('NDArray')
              ? 'NDArray'
              : desc.startsWith('Expr')
              ? 'Expr'
              : desc.startsWith('SymbolicMatrix')
              ? 'SymbolicMatrix'
              : desc.startsWith('FlintRationalPoly')
              ? 'FlintRationalPoly'
              : desc.split(' -> ').first,
          'summary': desc,
        });
      } catch (_) {}
    }
    return results;
  }

  void _writeWorkspace() {
    final workspaceFile = _getWorkspaceFile();
    final buffer = StringBuffer();
    buffer.writeln(
      '// ignore_for_file: unused_import, unused_element, non_constant_identifier_names',
    );
    buffer.writeln('// Auto-generated workspace. Do not edit.');
    final defaultImports = {
      "import 'dart:math' as math;",
      "import 'package:notebook/src/kernel_helper.dart';",
      "import 'package:ndarray/ndarray.dart';",
      "import 'package:symbolic_dart/symbolic_dart.dart' hide sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, abs;",
      "import 'package:resource_scope/resource_scope.dart';",
      _gpuArrayDefaultImport,
      _gpuArrayJitDefaultImport,
    };
    for (final imp in defaultImports) {
      buffer.writeln(imp);
    }
    for (final imp in _imports) {
      final trimmedImp = imp.trim();
      if (!defaultImports.contains(trimmedImp) &&
          trimmedImp != "import 'package:gpuarray/gpuarray.dart';" &&
          trimmedImp != "import 'package:gpuarray/jit.dart';") {
        buffer.writeln(imp);
      }
    }
    buffer.writeln();
    for (final def in _definitions.values) {
      buffer.writeln(def);
      buffer.writeln();
    }
    final content = buffer.toString();
    if (!workspaceFile.parent.existsSync()) {
      workspaceFile.parent.createSync(recursive: true);
    }
    workspaceFile.writeAsStringSync(content);
    if (_lspClient != null) {
      final fileUri = p.toUri(workspaceFile.path).toString();
      _lspClient!.didChange(fileUri, content, ++_workspaceVersion);
    }
  }
}
