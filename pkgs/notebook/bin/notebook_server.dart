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

void main(List<String> args) async {
  print('=== Dart NDArray Notebook Web Server ===');

  final dartSdkPath = p.dirname(p.dirname(Platform.resolvedExecutable));
  final workspaceDir = p.dirname(p.dirname(Platform.script.toFilePath()));

  int port = 8080;
  String? notebookPath;
  for (final arg in args) {
    final parsedPort = int.tryParse(arg);
    if (parsedPort != null) {
      port = parsedPort;
    } else {
      notebookPath = arg;
    }
  }

  print('SDK Path: $dartSdkPath');
  print('Workspace Dir: $workspaceDir');
  if (notebookPath != null) {
    print('Notebook File: $notebookPath');
  }

  final server = NotebookServer(
    workspaceDir: workspaceDir,
    dartSdkPath: dartSdkPath,
    port: port,
    notebookPath: notebookPath,
  );

  try {
    await server.start();
    print(
      '\n🚀 Notebook is live! Open http://localhost:$port in your browser.',
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
