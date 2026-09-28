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
import 'package:test/test.dart';
import 'package:notebook/notebook.dart';

void main() {
  late NotebookKernel kernel;

  setUp(() async {
    final workspaceDir = Directory.current.path;
    final sdkPath =
        Platform.environment['DART_SDK'] ??
        p.dirname(p.dirname(Platform.resolvedExecutable));

    kernel = NotebookKernel(workspaceDir: workspaceDir, dartSdkPath: sdkPath);
    await kernel.start();
  });

  tearDown(() async {
    await kernel.stop();
  });

  test('evaluates Image(NDArray) returning rendered BMP data URL', () async {
    await kernel.execute('''
var imgData = NDArray.fromList([
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 1.0, 0.0
], [2, 2, 3], DType.float64);
''');

    final result = await kernel.execute('Image(imgData)');
    expect(result, contains('data:image/bmp;base64,'));
  });

  test('evaluates 2D grayscale Image(NDArray)', () async {
    await kernel.execute('''
var grayData = NDArray.fromList([
  0.0, 0.5,
  0.8, 1.0
], [2, 2], DType.float64);
''');

    final result = await kernel.execute('Image(grayData)');
    expect(result, contains('data:image/bmp;base64,'));
  });
}
