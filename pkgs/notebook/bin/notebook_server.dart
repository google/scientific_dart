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
import 'package:path/path.dart' as p;
import 'package:notebook/src/notebook_server.dart';
import 'package:notebook/src/wasm_notebook_builder.dart';

void main(List<String> args) async {
  print('=== Dart NDArray Notebook Web Server ===');

  final dartSdkPath = p.dirname(p.dirname(Platform.resolvedExecutable));
  final workspaceDir = p.dirname(p.dirname(Platform.script.toFilePath()));
  final repoRoot = p.dirname(p.dirname(workspaceDir));

  var port = 8080;
  var serverlessWasm = false;
  String? staticBundleDir;
  String? notebookPath;

  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (arg == '--wasm' || arg == '--serverless-wasm') {
      serverlessWasm = true;
    } else if (arg == '--bundle-dir' && i + 1 < args.length) {
      staticBundleDir = args[++i];
    } else if (arg == '--port' && i + 1 < args.length) {
      final parsed = int.tryParse(args[++i]);
      if (parsed == null) {
        stderr.writeln('Invalid port: ${args[i]}');
        exitCode = 2;
        return;
      }
      port = parsed;
    } else {
      final parsed = int.tryParse(arg);
      if (parsed != null) {
        port = parsed;
      } else {
        notebookPath = arg;
      }
    }
  }

  print('SDK Path: $dartSdkPath');
  print('Workspace Dir: $workspaceDir');
  if (notebookPath != null) {
    print('Notebook File: $notebookPath');
  }

  if (serverlessWasm && staticBundleDir == null) {
    final defaultBundleDir = p.join(
      repoRoot,
      '.dart_tool',
      'wasm_notebook_bundle',
    );
    if (!File(p.join(defaultBundleDir, 'compiler_worker.wasm')).existsSync()) {
      print('Building serverless Wasm bundle...');
      final builder = WasmNotebookBuilder(
        workspaceRoot: repoRoot,
        dartSdkPath: dartSdkPath,
      );
      final result = await builder.buildStaticBundle(
        outputDir: defaultBundleDir,
      );
      staticBundleDir = result.outputDir;
    } else {
      staticBundleDir = defaultBundleDir;
    }
  }

  final server = NotebookServer(
    workspaceDir: workspaceDir,
    dartSdkPath: dartSdkPath,
    port: port,
    serverlessWasm: serverlessWasm,
    staticBundleDir: staticBundleDir,
    notebookPath: notebookPath,
  );

  try {
    await server.start();
    final modeSuffix = serverlessWasm ? '/?mode=wasm' : '';
    print(
      '\n🚀 Notebook is live! Open http://localhost:${server.actualPort}$modeSuffix in your browser.',
    );
    print('Press Ctrl+C or kill process to terminate.\n');

    // Keep process alive
    await ProcessSignal.sigint.watch().first;
  } catch (e, stack) {
    print('Error running notebook server: $e');
    print(stack);
  } finally {
    print('\nStopping server...');
    await server.stop();
    print('Goodbye.');
  }
}
