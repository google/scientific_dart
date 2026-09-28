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
        ? 'libndarray.dll'
        : (os == OS.macOS ? 'libndarray.dylib' : 'libndarray.so');

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    final compilerPath =
        cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');
    final compilerLower = compilerPath.toLowerCase();
    final isGNU =
        compilerLower.contains('gcc') ||
        compilerLower.contains('clang') ||
        compilerLower.contains('g++');
    final isMSVC = os == OS.windows && !isGNU;
    final msvcEnv = isMSVC ? await getMSVCEnvironment() : <String, String>{};

    // Compile highway if needed
    final highwayDir = input.packageRoot.resolve('third_party/highway/');
    final highwayBuildDir = Directory.fromUri(highwayDir.resolve('hwy_build'));

    final String hwyLibName;
    final String hwyContribLibName;
    final Uri hwyLibUri;
    final Uri hwyContribLibUri;

    if (isMSVC) {
      hwyLibName = 'hwy.lib';
      hwyContribLibName = 'hwy_contrib.lib';
      hwyLibUri = highwayBuildDir.uri.resolve('Release/$hwyLibName');
      hwyContribLibUri = highwayBuildDir.uri.resolve(
        'Release/$hwyContribLibName',
      );
    } else {
      hwyLibName = 'libhwy.a';
      hwyContribLibName = 'libhwy_contrib.a';
      hwyLibUri = highwayBuildDir.uri.resolve(hwyLibName);
      hwyContribLibUri = highwayBuildDir.uri.resolve(hwyContribLibName);
    }

    final libhwy = File.fromUri(hwyLibUri);
    final libhwyContrib = File.fromUri(hwyContribLibUri);

    if (!libhwy.existsSync() || !libhwyContrib.existsSync()) {
      print('Highway static libraries not found. Compiling highway...');
      if (!highwayBuildDir.existsSync()) {
        highwayBuildDir.createSync(recursive: true);
      }

      // Run cmake configuration
      final cmakeRes = await Process.run(
        'cmake',
        [
          '-DCMAKE_BUILD_TYPE=Release',
          '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
          '-DHWY_ENABLE_TESTS=OFF',
          '-DHWY_ENABLE_EXAMPLES=OFF',
          '..',
        ],
        workingDirectory: highwayBuildDir.path,
        environment: msvcEnv,
      );

      if (cmakeRes.exitCode != 0) {
        throw StateError(
          'CMake failed for highway (exit ${cmakeRes.exitCode}):\n'
          'stdout: ${cmakeRes.stdout}\n'
          'stderr: ${cmakeRes.stderr}',
        );
      }

      // Run cmake build
      final buildRes = await Process.run(
        'cmake',
        [
          '--build',
          '.',
          '--target',
          'hwy',
          'hwy_contrib',
          if (isMSVC) ...['--config', 'Release'],
          '--parallel',
        ],
        workingDirectory: highwayBuildDir.path,
        environment: msvcEnv,
      );

      if (buildRes.exitCode != 0) {
        throw StateError(
          'Build failed for highway (exit ${buildRes.exitCode}):\n'
          'stdout: ${buildRes.stdout}\n'
          'stderr: ${buildRes.stderr}',
        );
      }
      print('Highway compiled successfully.');
    }

    var cppCompilerPath = compilerPath;
    if (compilerPath.endsWith('gcc')) {
      cppCompilerPath =
          '${compilerPath.substring(0, compilerPath.length - 3)}g++';
    } else if (compilerPath.endsWith('clang')) {
      cppCompilerPath = '$compilerPath++';
    } else if (compilerPath.endsWith('cc')) {
      cppCompilerPath = 'c++';
    } else if (compilerPath.contains('gcc-')) {
      cppCompilerPath = compilerPath.replaceAll('gcc-', 'g++-');
    } else if (compilerPath.contains('clang-')) {
      cppCompilerPath = compilerPath.replaceAll('clang-', 'clang++-');
    }

    print(
      'Compiling ndarray custom C++ extensions using compiler: $cppCompilerPath',
    );

    final isX64 = input.config.code.targetArchitecture == Architecture.x64;

    if (isMSVC) {
      final ufuncsObj = outputDir.uri.resolve('custom_ufuncs.obj').toFilePath();
      final sortingObj = outputDir.uri
          .resolve('custom_sorting.obj')
          .toFilePath();
      final indexingObj = outputDir.uri
          .resolve('custom_indexing.obj')
          .toFilePath();
      final minizObj = outputDir.uri.resolve('miniz.obj').toFilePath();
      final npzIoObj = outputDir.uri.resolve('npz_io.obj').toFilePath();

      var res = await Process.run(cppCompilerPath, [
        '/c',
        '/O2',
        if (isX64) '/arch:AVX2',
        '/MD',
        '/EHsc',
        '/D_USE_MATH_DEFINES',
        '/I${input.packageRoot.toFilePath()}',
        input.packageRoot.resolve('hook/custom_ufuncs.cpp').toFilePath(),
        '/Fo:$ufuncsObj',
      ], environment: msvcEnv);
      if (res.exitCode != 0) {
        throw StateError(
          'Ufuncs compilation failed:\n'
          'stdout: ${res.stdout}\n'
          'stderr: ${res.stderr}',
        );
      }

      res = await Process.run(cppCompilerPath, [
        '/c',
        '/O2',
        if (isX64) '/arch:AVX2',
        '/MD',
        '/EHsc',
        '/D_USE_MATH_DEFINES',
        '/I${input.packageRoot.toFilePath()}',
        '/I${input.packageRoot.resolve('third_party/highway/').toFilePath()}',
        input.packageRoot.resolve('hook/custom_sorting.cpp').toFilePath(),
        '/Fo:$sortingObj',
      ], environment: msvcEnv);
      if (res.exitCode != 0) {
        throw StateError(
          'Sorting compilation failed:\n'
          'stdout: ${res.stdout}\n'
          'stderr: ${res.stderr}',
        );
      }

      res = await Process.run(cppCompilerPath, [
        '/c',
        '/O2',
        if (isX64) '/arch:AVX2',
        '/MD',
        '/EHsc',
        '/D_USE_MATH_DEFINES',
        '/I${input.packageRoot.toFilePath()}',
        input.packageRoot.resolve('hook/custom_indexing.cpp').toFilePath(),
        '/Fo:$indexingObj',
      ], environment: msvcEnv);
      if (res.exitCode != 0) {
        throw StateError(
          'Indexing compilation failed:\n'
          'stdout: ${res.stdout}\n'
          'stderr: ${res.stderr}',
        );
      }

      res = await Process.run(compilerPath, [
        '/c',
        '/O2',
        '/MD',
        '/I${input.packageRoot.toFilePath()}',
        input.packageRoot.resolve('third_party/miniz/miniz.c').toFilePath(),
        '/Fo:$minizObj',
      ], environment: msvcEnv);
      if (res.exitCode != 0) {
        throw StateError(
          'miniz compilation failed:\n'
          'stdout: ${res.stdout}\n'
          'stderr: ${res.stderr}',
        );
      }

      res = await Process.run(cppCompilerPath, [
        '/c',
        '/O2',
        if (isX64) '/arch:AVX2',
        '/MD',
        '/EHsc',
        '/I${input.packageRoot.toFilePath()}',
        input.packageRoot.resolve('hook/npz_io.cpp').toFilePath(),
        '/Fo:$npzIoObj',
      ], environment: msvcEnv);
      if (res.exitCode != 0) {
        throw StateError(
          'npz_io compilation failed:\n'
          'stdout: ${res.stdout}\n'
          'stderr: ${res.stderr}',
        );
      }

      final allExports = [
        ...extractExportsFromBindings(
          input.packageRoot
              .resolve('lib/src/ndarray_bindings.dart')
              .toFilePath(),
        ),
        ...extractExportsFromBindings(
          input.packageRoot
              .resolve('lib/src/ndarray_extensions_bindings.dart')
              .toFilePath(),
        ),
      ];

      final defFile = File(
        outputDir.uri.resolve('libndarray.def').toFilePath(),
      );
      final defContent = [
        'LIBRARY libndarray',
        'EXPORTS',
        ...allExports,
      ].join('\n');
      await defFile.writeAsString(defContent);

      res = await Process.run(cppCompilerPath, [
        '/LD',
        '/MD',
        ufuncsObj,
        sortingObj,
        indexingObj,
        minizObj,
        npzIoObj,
        libhwyContrib.path,
        libhwy.path,
        '/Fe:${libFile.path}',
        '/link',
        '/def:${defFile.path}',
      ], environment: msvcEnv);
      if (res.exitCode != 0) {
        throw StateError(
          'Linking failed:\n'
          'stdout: ${res.stdout}\n'
          'stderr: ${res.stderr}',
        );
      }
    } else {
      final ufuncsObj = outputDir.uri.resolve('custom_ufuncs.o').toFilePath();
      final sortingObj = outputDir.uri.resolve('custom_sorting.o').toFilePath();
      final indexingObj = outputDir.uri
          .resolve('custom_indexing.o')
          .toFilePath();
      final minizObj = outputDir.uri.resolve('miniz.o').toFilePath();
      final npzIoObj = outputDir.uri.resolve('npz_io.o').toFilePath();

      bool needsCompile(String src, String obj) {
        final srcF = File(src);
        final objF = File(obj);
        if (!objF.existsSync()) return true;
        final objTime = objF.lastModifiedSync();
        if (srcF.lastModifiedSync().isAfter(objTime)) return true;
        for (final header in [
          'hook/custom_indexing.h',
          'hook/custom_sorting.h',
          'hook/custom_ufuncs.h',
        ]) {
          final hF = File(input.packageRoot.resolve(header).toFilePath());
          if (hF.existsSync() && hF.lastModifiedSync().isAfter(objTime)) {
            return true;
          }
        }
        return false;
      }

      final ufuncsSrc = input.packageRoot
          .resolve('hook/custom_ufuncs.cpp')
          .toFilePath();
      if (needsCompile(ufuncsSrc, ufuncsObj)) {
        print('Compiling custom_ufuncs.cpp...');
        final res = await Process.run(cppCompilerPath, [
          '-c',
          '-fPIC',
          '-O3',
          if (isX64) ...['-mavx2', '-mfma'],
          '-fno-math-errno',
          '-I${input.packageRoot.toFilePath()}',
          ufuncsSrc,
          '-o',
          ufuncsObj,
        ]);
        if (res.exitCode != 0) {
          throw StateError('Ufuncs compilation failed: ${res.stderr}');
        }
      }

      final sortingSrc = input.packageRoot
          .resolve('hook/custom_sorting.cpp')
          .toFilePath();
      if (needsCompile(sortingSrc, sortingObj)) {
        print('Compiling custom_sorting.cpp...');
        final res = await Process.run(cppCompilerPath, [
          '-c',
          '-fPIC',
          '-O3',
          if (isX64) ...['-mavx2', '-mfma'],
          '-fno-math-errno',
          '-I${input.packageRoot.toFilePath()}',
          '-I${input.packageRoot.resolve('third_party/highway/').toFilePath()}',
          sortingSrc,
          '-o',
          sortingObj,
        ]);
        if (res.exitCode != 0) {
          throw StateError('Sorting compilation failed: ${res.stderr}');
        }
      }

      final indexingSrc = input.packageRoot
          .resolve('hook/custom_indexing.cpp')
          .toFilePath();
      if (needsCompile(indexingSrc, indexingObj)) {
        print('Compiling custom_indexing.cpp...');
        final res = await Process.run(cppCompilerPath, [
          '-c',
          '-fPIC',
          '-O3',
          if (isX64) ...['-mavx2', '-mfma'],
          '-fno-math-errno',
          '-I${input.packageRoot.toFilePath()}',
          indexingSrc,
          '-o',
          indexingObj,
        ]);
        if (res.exitCode != 0) {
          throw StateError('Indexing compilation failed: ${res.stderr}');
        }
      }

      final minizSrc = input.packageRoot
          .resolve('third_party/miniz/miniz.c')
          .toFilePath();
      final minizH = input.packageRoot
          .resolve('third_party/miniz/miniz.h')
          .toFilePath();
      if (needsCompile(minizSrc, minizObj) || needsCompile(minizH, minizObj)) {
        print('Compiling miniz.c...');
        final res = await Process.run(compilerPath, [
          '-c',
          '-fPIC',
          '-O3',
          '-I${input.packageRoot.toFilePath()}',
          minizSrc,
          '-o',
          minizObj,
        ]);
        if (res.exitCode != 0) {
          throw StateError('miniz compilation failed: ${res.stderr}');
        }
      }

      final npzIoSrc = input.packageRoot
          .resolve('hook/npz_io.cpp')
          .toFilePath();
      if (needsCompile(npzIoSrc, npzIoObj) || needsCompile(minizH, npzIoObj)) {
        print('Compiling npz_io.cpp...');
        final res = await Process.run(cppCompilerPath, [
          '-c',
          '-fPIC',
          '-O3',
          '-I${input.packageRoot.toFilePath()}',
          npzIoSrc,
          '-o',
          npzIoObj,
        ]);
        if (res.exitCode != 0) {
          throw StateError('npz_io compilation failed: ${res.stderr}');
        }
      }

      final objs = [ufuncsObj, sortingObj, indexingObj, minizObj, npzIoObj];
      final needsLink =
          !libFile.existsSync() ||
          objs.any(
            (obj) =>
                File(obj)
                    .lastModifiedSync()
                    .isAfter(libFile.lastModifiedSync()),
          );

      if (needsLink) {
        print('Linking shared library...');
        final res = await Process.run(cppCompilerPath, [
          '-shared',
          '-fPIC',
          ufuncsObj,
          sortingObj,
          indexingObj,
          minizObj,
          npzIoObj,
          libhwyContrib.path,
          libhwy.path,
          '-o',
          libFile.path,
          if (os != OS.windows) '-lm',
        ]);
        if (res.exitCode != 0) {
          throw StateError('Linking failed: ${res.stderr}');
        }
      }
    }
    print('Compiled ndarray shared library successfully at: ${libFile.path}');

    // Register the dynamic CodeAsset in the hooks pipeline under package asset ID namespace
    if (libFile.existsSync()) {
      output.assets.code.add(
        CodeAsset(
          package: packageName,
          name: 'ndarray',
          linkMode: DynamicLoadingBundled(),
          file: libFile.uri,
        ),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_sorting.cpp'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_sorting.h'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_ufuncs.cpp'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_ufuncs.h'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_indexing.cpp'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_indexing.h'),
      );
      output.dependencies.add(input.packageRoot.resolve('hook/npz_io.cpp'));
      output.dependencies.add(input.packageRoot.resolve('hook/npz_io.h'));
      output.dependencies.add(
        input.packageRoot.resolve('third_party/miniz/miniz.c'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('third_party/miniz/miniz.h'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('third_party/timsort/timsort.h'),
      );
      print(
        'Registered ndarray native custom extensions code asset successfully.',
      );
    }
  });
}

List<String> extractExportsFromBindings(String bindingsPath) {
  final file = File(bindingsPath);
  if (!file.existsSync()) {
    print(
      'WARNING: Bindings file not found at $bindingsPath. Dynamic library exports list might be incomplete!',
    );
    return [];
  }

  final content = file.readAsStringSync();
  final regex = RegExp(r'external\s+[\w\d_<>.]+\s+(\w+)\s*\(');

  final exports = <String>[];
  for (final match in regex.allMatches(content)) {
    final name = match.group(1);
    if (name != null && !exports.contains(name)) {
      exports.add(name);
    }
  }
  print('Extracted ${exports.length} export symbols from $bindingsPath');
  return exports;
}

Future<Map<String, String>> getMSVCEnvironment() async {
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
