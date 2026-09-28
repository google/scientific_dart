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

import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final srcDir = Directory.fromUri(input.packageRoot.resolve('hook/src/'));
    if (!srcDir.existsSync()) {
      srcDir.createSync(recursive: true);
    }

    // 1. Download and extract KissFFT if not present
    final logHeader = File.fromUri(srcDir.uri.resolve('kiss_fft_log.h'));
    if (!logHeader.existsSync()) {
      print('Downloading KissFFT source files archive from GitHub...');
      final client = HttpClient();
      try {
        final request = await client.getUrl(
          Uri.parse(
            'https://github.com/mborgerding/kissfft/archive/6e9e673e420c4bf47d4a60c57c578f93e4ec192f.tar.gz',
          ),
        );
        final response = await request.close();
        if (response.statusCode != 200) {
          throw HttpException(
            'Failed to download KissFFT archive: status ${response.statusCode}',
          );
        }

        final bytesBuilder = BytesBuilder();
        await for (final chunk in response) {
          bytesBuilder.add(chunk);
        }
        final tarGzBytes = bytesBuilder.toBytes();

        final unzippedBytes = GZipDecoder().decodeBytes(tarGzBytes);
        final archive = TarDecoder().decodeBytes(unzippedBytes);

        for (final file in archive) {
          if (file.isFile) {
            final baseName = file.name.split('/').last;
            if (baseName.endsWith('.c') || baseName.endsWith('.h')) {
              final outFile = File.fromUri(srcDir.uri.resolve(baseName));
              outFile.writeAsBytesSync(file.content as List<int>, flush: true);
              print('Extracted: $baseName');
            }
          }
        }
      } finally {
        client.close();
      }
    }

    // 2. Get cross-compiler or host compiler from modern input config
    final packageName = input.packageName;
    final os = input.config.code.targetOS;
    final cCompiler = input.config.code.cCompiler;

    final libName = os == OS.windows
        ? 'libpocketfft.dll'
        : (os == OS.macOS ? 'libpocketfft.dylib' : 'libpocketfft.so');

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    // 3. Compile plain-C source code using Process.run for extreme reliability
    final compilerPath =
        cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');
    print('Compiling pocketfft plain C files via compiler: $compilerPath');

    final compilerLower = compilerPath.toLowerCase();
    final isGNU =
        compilerLower.contains('gcc') ||
        compilerLower.contains('clang') ||
        compilerLower.contains('g++');
    final isMSVC = os == OS.windows && !isGNU;
    final compileArgs = isMSVC
        ? <String>[
            '/LD',
            '/O2',
            '/EHsc',
            '/Dkiss_fft_scalar=double',
            '/I',
            srcDir.uri.toFilePath(),
            srcDir.uri.resolve('kiss_fft.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftr.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftnd.c').toFilePath(),
            '/Fe:${libFile.path}',
            '/link',
            '/EXPORT:kiss_fft_alloc',
            '/EXPORT:kiss_fft',
            '/EXPORT:kiss_fftr_alloc',
            '/EXPORT:kiss_fftr',
            '/EXPORT:kiss_fftri',
            '/EXPORT:kiss_fftnd_alloc',
            '/EXPORT:kiss_fftnd',
          ]
        : <String>[
            '-shared',
            '-fPIC',
            '-O3',
            '-ffast-math',
            '-Dkiss_fft_scalar=double',
            '-I',
            srcDir.uri.toFilePath(),
            srcDir.uri.resolve('kiss_fft.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftr.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftnd.c').toFilePath(),
            '-o',
            libFile.path,
            if (os != OS.windows) '-lm',
          ];

    print(
      'Environment Keys: ${Platform.environment.keys.where((k) => k.toUpperCase().contains("INC") || k.toUpperCase().contains("LIB") || k.toUpperCase() == "PATH").toList()}',
    );
    print(
      'Environment PATH: ${Platform.environment['PATH'] ?? Platform.environment['Path'] ?? Platform.environment['path']}',
    );
    print(
      'Environment INCLUDE: ${Platform.environment['INCLUDE'] ?? Platform.environment['Include'] ?? Platform.environment['include']}',
    );
    print(
      'Environment LIB: ${Platform.environment['LIB'] ?? Platform.environment['Lib'] ?? Platform.environment['lib']}',
    );

    final runEnv = <String, String>{...Platform.environment};
    if (isMSVC) {
      final msvcEnv = await getMSVCEnvironment();
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
      throw StateError(
        'PocketFFT native C compilation failed (exit ${res.exitCode}):\n'
        'stdout: ${res.stdout}\n'
        'stderr: ${res.stderr}',
      );
    }
    print('Compiled shared library binary successfully at: ${libFile.path}');

    // 4. Register the dynamic CodeAsset in the hooks pipeline
    if (libFile.existsSync()) {
      output.assets.code.add(
        CodeAsset(
          package: packageName,
          name: 'pocketfft',
          linkMode: DynamicLoadingBundled(),
          file: libFile.uri,
        ),
      );
      output.dependencies.add(srcDir.uri.resolve('kiss_fft.c'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftr.c'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftnd.c'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fft.h'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fft_log.h'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftnd.h'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftr.h'));
      output.dependencies.add(srcDir.uri.resolve('_kiss_fft_guts.h'));
      print('Registered pocketfft native dynamic code asset successfully.');
    }
  });
}

/// Helper function to query Visual Studio to obtain the proper environment variables
/// (like INCLUDE, LIB, and LIBPATH) for MSVC compilation on Windows.
Future<Map<String, String>> getMSVCEnvironment() async {
  if (!Platform.isWindows) return {};

  // Find vswhere.exe
  String vswherePath = 'vswhere.exe'; // Try PATH first
  // Fallback to default installer directory if not in PATH
  final programFilesX86 =
      Platform.environment['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
  final defaultVswhere =
      '$programFilesX86\\Microsoft Visual Studio\\Installer\\vswhere.exe';
  if (await File(defaultVswhere).exists()) {
    vswherePath = defaultVswhere;
  }

  try {
    // Run vswhere to find Visual Studio installation path
    final vswhereRes = await Process.run(vswherePath, [
      '-latest',
      '-property',
      'installationPath',
    ]);
    if (vswhereRes.exitCode != 0) {
      print('vswhere failed with exit code ${vswhereRes.exitCode}');
      return {};
    }

    final vsPath = vswhereRes.stdout.toString().trim();
    if (vsPath.isEmpty) {
      print('vswhere returned empty path');
      return {};
    }

    final vcvarsPath = '$vsPath\\VC\\Auxiliary\\Build\\vcvarsall.bat';
    if (!await File(vcvarsPath).exists()) {
      print('vcvarsall.bat not found at $vcvarsPath');
      return {};
    }

    // To avoid Dart process argument escaping issues on Windows, we write a temporary
    // batch file that calls vcvarsall.bat and prints the environment, then run it.
    final tempDir = Directory.systemTemp;
    final tempFile = File(
      '${tempDir.path}\\get_msvc_env_${DateTime.now().millisecondsSinceEpoch}.bat',
    );
    try {
      await tempFile.writeAsString(
        '@echo off\ncall "$vcvarsPath" amd64\nset\n',
      );
    } catch (e) {
      print('Failed to write temporary batch file: $e');
      return {};
    }

    final envRes = await Process.run('cmd.exe', ['/c', tempFile.path]);

    try {
      await tempFile.delete();
    } catch (_) {}

    if (envRes.exitCode != 0) {
      print(
        'Temporary MSVC environment batch file failed with exit code ${envRes.exitCode}',
      );
      print('vcvarsall.bat stdout: ${envRes.stdout}');
      print('vcvarsall.bat stderr: ${envRes.stderr}');
      return {};
    }

    final envMap = <String, String>{};
    final lines = envRes.stdout.toString().split('\n');
    for (final line in lines) {
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
  } catch (e) {
    print('Error detecting MSVC environment: $e');
    return {};
  }
}
