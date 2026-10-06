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

import 'package:notebook/src/notebook_server.dart';
import 'package:notebook/src/wasm_notebook_builder.dart';
import 'package:path/path.dart' as p;

Future<void> main(List<String> args) async {
  var outputDir = '';
  var forceRebuild = false;
  var serve = false;
  var port = 8080;

  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (arg == '--help' || arg == '-h') {
      _printUsage();
      return;
    } else if (arg == '--serve') {
      serve = true;
    } else if (arg == '--force-rebuild') {
      forceRebuild = true;
    } else if (arg == '--output' || arg == '-o') {
      if (i + 1 >= args.length) {
        stderr.writeln('Missing value for $arg.');
        _printUsage();
        exitCode = 2;
        return;
      }
      outputDir = args[++i];
    } else if (arg.startsWith('--output=')) {
      outputDir = arg.substring('--output='.length);
    } else if (arg == '--port' || arg == '-p') {
      if (i + 1 >= args.length) {
        stderr.writeln('Missing value for $arg.');
        _printUsage();
        exitCode = 2;
        return;
      }
      final parsedPort = int.tryParse(args[++i]);
      if (parsedPort == null || parsedPort < 0 || parsedPort > 65535) {
        stderr.writeln('Invalid port number: ${args[i]}');
        exitCode = 2;
        return;
      }
      port = parsedPort;
    } else if (arg.startsWith('--port=')) {
      final parsedPort = int.tryParse(arg.substring('--port='.length));
      if (parsedPort == null || parsedPort < 0 || parsedPort > 65535) {
        stderr.writeln('Invalid port number: $arg');
        exitCode = 2;
        return;
      }
      port = parsedPort;
    } else {
      stderr.writeln('Unknown option: $arg');
      _printUsage();
      exitCode = 2;
      return;
    }
  }

  final dartSdkPath = p.dirname(p.dirname(Platform.resolvedExecutable));
  final notebookPkgDir = p.dirname(p.dirname(Platform.script.toFilePath()));
  final repoRoot = p.dirname(p.dirname(notebookPkgDir));

  print('=== Building Serverless Wasm Notebook Bundle ===');
  print('Repo root: $repoRoot');
  print('Dart SDK:  $dartSdkPath');

  final builder = WasmNotebookBuilder(
    workspaceRoot: repoRoot,
    dartSdkPath: dartSdkPath,
  );

  final result = await builder.buildStaticBundle(
    outputDir: outputDir.isEmpty ? null : outputDir,
    forceRebuildCompiler: forceRebuild,
  );

  print('\nStatic Wasm notebook bundle staged at: ${result.outputDir}');
  print(
    '  compiler_worker.wasm: ${(result.compilerWorkerWasmBytes / (1024 * 1024)).toStringAsFixed(2)} MB',
  );
  print(
    '  native_math.wasm:     ${(result.nativeMathWasmBytes / 1024).toStringAsFixed(1)} KB',
  );
  print('  Bundled source files: ${result.bundledSourceFileCount}');
  print('  DartPad LSP assets:   ${result.hasDartPadWorkerAssets}');

  if (serve) {
    final server = NotebookServer(
      workspaceDir: notebookPkgDir,
      dartSdkPath: dartSdkPath,
      port: port,
      serverlessWasm: true,
      staticBundleDir: result.outputDir,
    );
    try {
      await server.start();
      print(
        '\nServerless Wasm Notebook running at '
        'http://localhost:${server.actualPort}/?mode=wasm',
      );
      print('Press Ctrl+C to stop.');
      await ProcessSignal.sigint.watch().first;
    } finally {
      await server.stop();
    }
  }
}

void _printUsage() {
  stdout.writeln(
    'Usage: dart run pkgs/notebook/bin/build_wasm_notebook.dart [options]\n'
    '\n'
    'Options:\n'
    '  -o, --output <dir>     Output directory for the static bundle\n'
    '      --force-rebuild    Force recompilation of compiler_worker.wasm\n'
    '      --serve            Serve the static bundle over HTTP (no active VM kernel)\n'
    '  -p, --port <port>      HTTP port when --serve is passed (default: 8080)\n'
    '  -h, --help             Show this usage information',
  );
}
