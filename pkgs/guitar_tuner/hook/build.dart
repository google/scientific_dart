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
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final packageName = input.packageName;
    final os = input.config.code.targetOS;
    final cCompiler = input.config.code.cCompiler;

    final libName = os == OS.windows
        ? 'libtuner.dll'
        : (os == OS.macOS ? 'libtuner.dylib' : 'libtuner.so');

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    final compilerPath = cCompiler?.compiler.toFilePath() ?? 'cc';

    final isMSVC =
        os == OS.windows && compilerPath.toLowerCase().contains('cl');

    final compileArgs = isMSVC
        ? <String>[
            '/LD',
            '/O2',
            '/EHsc',
            input.packageRoot.resolve('hook/tuner_bridge.c').toFilePath(),
            '/Fe:${libFile.path}',
          ]
        : <String>[
            '-shared',
            '-fPIC',
            '-O3',
            input.packageRoot.resolve('hook/tuner_bridge.c').toFilePath(),
            '-o',
            libFile.path,
            '-lm',
          ];

    if (!isMSVC) {
      if (os == OS.linux) {
        compileArgs.add('-lpthread');
        compileArgs.add('-ldl');
      } else if (os == OS.macOS) {
        compileArgs.addAll([
          '-Wl,-headerpad_max_install_names',
          '-framework',
          'CoreAudio',
          '-framework',
          'AudioToolbox',
          '-framework',
          'AudioUnit',
          '-framework',
          'CoreFoundation',
        ]);
      }
    }

    final res = await Process.run(compilerPath, compileArgs);
    if (res.exitCode != 0) {
      throw StateError('Guitar tuner bridge compilation failed: ${res.stderr}');
    }

    if (libFile.existsSync()) {
      output.assets.code.add(
        CodeAsset(
          package: packageName,
          name: 'tuner_bridge',
          linkMode: DynamicLoadingBundled(),
          file: libFile.uri,
        ),
      );
      output.dependencies.add(input.packageRoot.resolve('hook/tuner_bridge.c'));
    }
  });
}
