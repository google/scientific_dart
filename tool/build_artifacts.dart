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
    ..addOption(
      'ios-sdk',
      allowed: IOSSdk.values.map((s) => s.type),
      help:
          'Target iOS SDK (`iphoneos` or `iphonesimulator`, required for `--os ios`).',
    )
    ..addOption(
      'android-ndk-api',
      defaultsTo: '24',
      help: 'Minimum Android NDK API level when `--os android` is selected.',
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
  final rawIosSdk = parsed.option('ios-sdk');
  final IOSSdk? targetIosSdk;
  if (targetOS == OS.iOS) {
    if (rawIosSdk == null) {
      stderr.writeln('Error: `--ios-sdk` is required when `--os ios` is set.');
      exit(64);
    }
    targetIosSdk = IOSSdk.fromString(rawIosSdk);
  } else {
    if (rawIosSdk != null) {
      stderr.writeln(
        'Error: `--ios-sdk` is only valid when `--os ios` is set.',
      );
      exit(64);
    }
    targetIosSdk = null;
  }
  final androidNdkApi = int.tryParse(parsed.option('android-ndk-api')!);
  if (androidNdkApi == null || androidNdkApi < 24) {
    stderr.writeln('Error: `--android-ndk-api` must be an integer >= 24.');
    exit(64);
  }

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
  final targetTag = targetIosSdk != null
      ? '${targetOS.name}-${targetIosSdk.type}-${targetArch.name}'
      : '${targetOS.name}-${targetArch.name}';

  final builtFiles = <File>[];

  for (final pkg in packages) {
    stdout.writeln('=== Building $pkg for $targetTag ===');
    final packageRoot = workspaceRoot.resolve('pkgs/$pkg/');
    final buildBaseDir = Directory.fromUri(
      workspaceRoot.resolve('.dart_tool/artifact_builder/$pkg/$targetTag/'),
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
          android: targetOS == OS.android
              ? AndroidCodeConfig(targetNdkApi: androidNdkApi)
              : null,
          iOS: targetOS == OS.iOS
              ? IOSCodeConfig(targetSdk: targetIosSdk!, targetVersion: 13)
              : null,
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
      final artifactName = '$shortName-$targetTag.$ext';
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
  if (targetOS == OS.android) {
    return _detectAndroidNdkCompiler();
  }
  if (targetOS == OS.iOS && Platform.isMacOS) {
    final clangCheck = await Process.run('xcrun', ['--find', 'clang']);
    final arCheck = await Process.run('xcrun', ['--find', 'ar']);
    final ldCheck = await Process.run('xcrun', ['--find', 'ld']);
    if (clangCheck.exitCode == 0 &&
        arCheck.exitCode == 0 &&
        ldCheck.exitCode == 0) {
      return CCompilerConfig(
        compiler: Uri.file(clangCheck.stdout.toString().trim()),
        archiver: Uri.file(arCheck.stdout.toString().trim()),
        linker: Uri.file(ldCheck.stdout.toString().trim()),
      );
    }
  }
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

CCompilerConfig? _detectAndroidNdkCompiler() {
  final ndkCandidates = <Directory>[];
  for (final key in const [
    'ANDROID_NDK_LATEST_HOME',
    'ANDROID_NDK_ROOT',
    'ANDROID_NDK_HOME',
    'ANDROID_NDK',
  ]) {
    final val = Platform.environment[key];
    if (val != null && val.isNotEmpty) {
      final dir = Directory(val);
      if (dir.existsSync()) ndkCandidates.add(dir);
    }
  }
  for (final sdkKey in const ['ANDROID_HOME', 'ANDROID_SDK_ROOT']) {
    final sdkVal = Platform.environment[sdkKey];
    if (sdkVal == null || sdkVal.isEmpty) continue;
    final ndkParent = Directory('$sdkVal/ndk');
    if (ndkParent.existsSync()) {
      final versions = ndkParent.listSync().whereType<Directory>().toList()
        ..sort(_compareVersionDesc);
      ndkCandidates.addAll(versions);
    }
  }
  final ext = Platform.isWindows ? '.exe' : '';
  for (final ndkDir in ndkCandidates) {
    final prebuilt = Directory.fromUri(
      ndkDir.uri.resolve('toolchains/llvm/prebuilt/'),
    );
    if (!prebuilt.existsSync()) continue;
    for (final hostDir in prebuilt.listSync().whereType<Directory>()) {
      final clang = File.fromUri(hostDir.uri.resolve('bin/clang$ext'));
      final ar = File.fromUri(hostDir.uri.resolve('bin/llvm-ar$ext'));
      final ld = File.fromUri(hostDir.uri.resolve('bin/ld.lld$ext'));
      if (clang.existsSync() && ar.existsSync() && ld.existsSync()) {
        return CCompilerConfig(
          compiler: clang.uri,
          archiver: ar.uri,
          linker: ld.uri,
        );
      }
    }
  }
  throw StateError(
    'Android NDK toolchain not found. Set ANDROID_NDK_LATEST_HOME, '
    'ANDROID_NDK_ROOT, ANDROID_NDK_HOME, or ANDROID_HOME.',
  );
}

int _compareVersionDesc(Directory a, Directory b) {
  final aName = a.uri.pathSegments.where((s) => s.isNotEmpty).last;
  final bName = b.uri.pathSegments.where((s) => s.isNotEmpty).last;
  final aParts = aName.split('.').map(int.tryParse).toList();
  final bParts = bName.split('.').map(int.tryParse).toList();
  final len = aParts.length > bParts.length ? aParts.length : bParts.length;
  for (var i = 0; i < len; i++) {
    final av = i < aParts.length ? aParts[i] : 0;
    final bv = i < bParts.length ? bParts[i] : 0;
    if (av == null || bv == null) return bName.compareTo(aName);
    if (av != bv) return bv.compareTo(av);
  }
  return bName.compareTo(aName);
}
