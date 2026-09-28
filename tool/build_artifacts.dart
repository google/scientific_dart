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

// Builds precompiled native dynamic libraries for scientific_dart packages
// (`pocketfft`, `openblas`, `ndarray`, `symbolic_dart`) by invoking their
// `hook/build.dart` in `source` mode and packaging the resulting binaries
// into a release artifact directory (`dist/`) with `SHA256SUMS.txt`.

import 'dart:convert';
import 'dart:io';

import 'package:args/args.dart';
import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';

const _defaultPackages = ['pocketfft', 'openblas', 'ndarray'];
const _allSupportedPackages = [
  'pocketfft',
  'openblas',
  'ndarray',
  'symbolic_dart',
];

Future<void> main(List<String> args) async {
  final parser = ArgParser()
    ..addOption(
      'os',
      abbr: 'o',
      allowed: OS.values.map((o) => o.name),
      defaultsTo: OS.current.name,
      help: 'Target operating system.',
    )
    ..addOption(
      'architecture',
      abbr: 'a',
      allowed: Architecture.values.map((a) => a.name),
      defaultsTo: Architecture.current.name,
      help: 'Target CPU architecture.',
    )
    ..addMultiOption(
      'package',
      abbr: 'p',
      allowed: _allSupportedPackages,
      defaultsTo: _defaultPackages,
      help: 'Packages to build native artifacts for.',
    )
    ..addOption(
      'output-dir',
      abbr: 'd',
      defaultsTo: 'dist',
      help: 'Output directory for release binary artifacts.',
    )
    ..addFlag(
      'help',
      abbr: 'h',
      negatable: false,
      help: 'Show usage information.',
    );

  final ArgResults parsed;
  try {
    parsed = parser.parse(args);
  } catch (e) {
    stderr.writeln('Error: $e\n\n${parser.usage}');
    exit(64);
  }

  if (parsed.flag('help')) {
    stdout.writeln('Usage: dart tool/build_artifacts.dart [options]\n');
    stdout.writeln(parser.usage);
    return;
  }

  final targetOS = OS.fromString(parsed.option('os')!);
  final targetArch = Architecture.fromString(parsed.option('architecture')!);
  final packages = parsed.multiOption('package');
  final workspaceRoot = Directory.current.uri;
  final outDir = Directory.fromUri(
    workspaceRoot.resolve('${parsed.option('output-dir')!}/'),
  );
  await outDir.create(recursive: true);

  final ext = switch (targetOS) {
    OS.windows => 'dll',
    OS.macOS || OS.iOS => 'dylib',
    _ => 'so',
  };

  final builtFiles = <File>[];

  for (final pkg in packages) {
    stdout.writeln(
      '=== Building $pkg for ${targetOS.name}-${targetArch.name} ===',
    );
    final packageRoot = workspaceRoot.resolve('pkgs/$pkg/');
    final buildBaseDir = Directory.fromUri(
      workspaceRoot.resolve(
        '.dart_tool/artifact_builder/$pkg/${targetOS.name}-${targetArch.name}/',
      ),
    );
    final sharedDir = Directory.fromUri(buildBaseDir.uri.resolve('shared/'));
    await sharedDir.create(recursive: true);
    final outputFile = File.fromUri(buildBaseDir.uri.resolve('output.json'));
    final inputFile = File.fromUri(buildBaseDir.uri.resolve('input.json'));

    final cCompilerConfig = await _detectCrossCompiler(targetOS, targetArch);

    final buildInputBuilder = BuildInputBuilder()
      ..setupShared(
        packageRoot: packageRoot,
        packageName: pkg,
        outputDirectoryShared: sharedDir.uri,
        outputFile: outputFile.uri,
        userDefines: PackageUserDefines(
          workspacePubspec: PackageUserDefinesSource(
            defines: const {'buildMode': 'source'},
            basePath: workspaceRoot,
          ),
        ),
      )
      ..config.setupBuild(linkingEnabled: false)
      ..addExtension(
        CodeAssetExtension(
          targetOS: targetOS,
          targetArchitecture: targetArch,
          linkModePreference: LinkModePreference.dynamic,
          cCompiler: cCompilerConfig,
          macOS: targetOS == OS.macOS
              ? MacOSCodeConfig(targetVersion: 13)
              : null,
        ),
      );

    await inputFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(buildInputBuilder.json),
    );

    final hookScript = packageRoot
        .resolve('hook/build.dart')
        .toFilePath(windows: Platform.isWindows);
    final res = await Process.run(
      Platform.resolvedExecutable,
      [hookScript, '--config=${inputFile.path}'],
      workingDirectory: workspaceRoot.toFilePath(windows: Platform.isWindows),
      environment: {
        ...Platform.environment,
        '${pkg.toUpperCase()}_BUILD_MODE': 'source',
        'SCIENTIFIC_DART_BUILD_MODE': 'source',
      },
    );

    if ((res.stdout as String).isNotEmpty) {
      stdout.write(res.stdout);
    }
    if ((res.stderr as String).isNotEmpty) {
      stderr.write(res.stderr);
    }
    if (res.exitCode != 0) {
      throw StateError(
        'Hook build failed for package:$pkg (exit ${res.exitCode}).',
      );
    }

    final outputJson =
        jsonDecode(await outputFile.readAsString()) as Map<String, Object?>;
    final buildOutput = BuildOutput(outputJson);

    for (final codeAsset in buildOutput.assets.code) {
      final assetFileUri = codeAsset.file;
      if (assetFileUri == null) continue;
      final srcFile = File.fromUri(assetFileUri);
      if (!srcFile.existsSync()) {
        throw StateError(
          'Expected built asset file not found at ${srcFile.path}',
        );
      }

      // Asset IDs have the form `package:<pkg>/<assetName>`
      final shortName = codeAsset.id.split('/').last;
      if (shortName == 'ndarray_cpu_check') continue;
      final artifactName =
          '$shortName-${targetOS.name}-${targetArch.name}.$ext';
      final dstFile = File.fromUri(outDir.uri.resolve(artifactName));
      await srcFile.copy(dstFile.path);
      builtFiles.add(dstFile);
      stdout.writeln('  -> Packaged $artifactName (${dstFile.lengthSync()} B)');
    }
  }

  final sumsFile = File.fromUri(outDir.uri.resolve('SHA256SUMS.txt'));
  final existingLines = <String, String>{};
  if (sumsFile.existsSync()) {
    for (final line in await sumsFile.readAsLines()) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;
      final parts = trimmed.split(RegExp(r'\s+'));
      if (parts.length >= 2) {
        existingLines[parts.last] = parts.first;
      }
    }
  }

  for (final file in builtFiles) {
    final name = file.uri.pathSegments.last;
    final digest = sha256.convert(await file.readAsBytes()).toString();
    existingLines[name] = digest;
  }

  final sortedEntries = existingLines.entries.toList()
    ..sort((a, b) => a.key.compareTo(b.key));
  await sumsFile.writeAsString(
    '${sortedEntries.map((e) => '${e.value}  ${e.key}').join('\n')}\n',
  );
  stdout.writeln(
    'Updated ${sumsFile.path} with ${sortedEntries.length} hashes.',
  );
}

Future<CCompilerConfig?> _detectCrossCompiler(
  OS targetOS,
  Architecture targetArch,
) async {
  if (targetOS == OS.linux &&
      targetArch == Architecture.arm64 &&
      Architecture.current != Architecture.arm64) {
    final gccCheck = await Process.run('which', ['aarch64-linux-gnu-gcc']);
    final arCheck = await Process.run('which', ['aarch64-linux-gnu-ar']);
    final ldCheck = await Process.run('which', ['aarch64-linux-gnu-ld']);
    if (gccCheck.exitCode == 0 &&
        arCheck.exitCode == 0 &&
        ldCheck.exitCode == 0) {
      return CCompilerConfig(
        compiler: Uri.file(gccCheck.stdout.toString().trim()),
        archiver: Uri.file(arCheck.stdout.toString().trim()),
        linker: Uri.file(ldCheck.stdout.toString().trim()),
      );
    }
  }
  return null;
}
