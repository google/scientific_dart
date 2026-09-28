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
import 'package:crypto/crypto.dart';
import 'package:gpuarray/src/hook_helpers/build_options.dart';
import 'package:gpuarray/src/hook_helpers/hashes.dart';
import 'package:hooks/hooks.dart';

Future<void> main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final BuildOptions buildOptions;
    try {
      buildOptions = BuildOptions.fromDefines(input.userDefines);
    } on Object catch (error) {
      stderr.writeln(BuildOptions.usageError(error));
      exitCode = 2;
      return;
    }

    final buildMode = switch (buildOptions.buildMode) {
      BuildModeEnum.fetch => FetchMode(input),
      BuildModeEnum.local => LocalMode(input, buildOptions.localPath),
      BuildModeEnum.source => SourceMode(input, buildOptions.checkoutPath),
    };

    final Uri? builtLibrary;
    try {
      builtLibrary = await buildMode.build();
    } on FormatException catch (error) {
      stderr.writeln(BuildOptions.usageError(error));
      exitCode = 2;
      return;
    }
    if (builtLibrary == null) {
      return;
    }

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'wgpu_native',
        linkMode: DynamicLoadingBundled(),
        file: builtLibrary,
      ),
    );
    output.dependencies.addAll(buildMode.dependencies);
    output.dependencies.add(input.packageRoot.resolve('pubspec.yaml'));
  });
}

/// Base build mode strategy for resolving `wgpu-native` native code assets.
sealed class BuildMode {
  /// The hook build input configuration.
  final BuildInput input;

  /// Creates a [BuildMode] for [input].
  const BuildMode(this.input);

  /// Additional file dependencies to register with the build hook output.
  List<Uri> get dependencies;

  /// Resolves or builds the `wgpu-native` shared library and returns its URI.
  Future<Uri?> build();
}

/// Downloads and verifies the prebuilt `wgpu-native` release archive.
final class FetchMode extends BuildMode {
  /// Creates a [FetchMode] for [input].
  const FetchMode(super.input);

  @override
  Future<Uri?> build() async {
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final asset = fileHashes[(os, arch)];

    if (asset == null) {
      stderr.writeln(
        'Warning: No prebuilt wgpu-native release configuration for $os $arch. '
        'Hardware acceleration will fall back to CPU simulation mode.',
      );
      return null;
    }

    final extractDirectory = Directory.fromUri(
      input.outputDirectoryShared.resolve(
        'wgpu-native-$wgpuVersion/${os.name}-${arch.name}/',
      ),
    );
    if (!extractDirectory.existsSync()) {
      extractDirectory.createSync(recursive: true);
    }

    final libraryFile = File(
      extractDirectory.uri.resolve(asset.libName).toFilePath(),
    );
    if (!libraryFile.existsSync()) {
      final downloadUrl = Uri.parse('$wgpuBaseUrl/${asset.zipName}');
      final zipBytes = await _downloadWithRedirects(downloadUrl);

      final actualSha256 = sha256.convert(zipBytes).toString().toLowerCase();
      final expectedSha256 = asset.sha256.toLowerCase();
      if (actualSha256 != expectedSha256) {
        throw StateError(
          'Security Error: SHA-256 hash mismatch for ${asset.zipName}!\n'
          'Expected: $expectedSha256\n'
          'Actual:   $actualSha256',
        );
      }

      final archive = ZipDecoder().decodeBytes(zipBytes);
      for (final file in archive) {
        if (file.isFile) {
          final baseName = file.name.split('/').last;
          final outputFile = File(
            extractDirectory.uri.resolve(baseName).toFilePath(),
          );
          outputFile.writeAsBytesSync(file.content as List<int>, flush: true);
        }
      }
    }

    return libraryFile.existsSync() ? libraryFile.uri : null;
  }

  @override
  List<Uri> get dependencies => const [];
}

/// Copies a locally provided `wgpu-native` shared library binary.
final class LocalMode extends BuildMode {
  /// User-provided path to the local `wgpu-native` binary.
  final Uri? localPath;

  /// Creates a [LocalMode] for [input] and [localPath].
  const LocalMode(super.input, this.localPath);

  File _resolveLocalFile() {
    final path = localPath;
    if (path == null) {
      throw const FormatException(
        '`localPath` is not set in `hooks.user_defines.gpuarray` '
        '(or `LOCAL_GPUARRAY_BINARY` environment variable).',
      );
    }
    final file = File(path.toFilePath(windows: Platform.isWindows));
    if (!file.existsSync()) {
      throw FileSystemException(
        'Could not find local wgpu-native binary.',
        file.path,
      );
    }
    return file;
  }

  @override
  Future<Uri?> build() async {
    final sourceFile = _resolveLocalFile();
    final destinationFile = File.fromUri(
      input.outputDirectory.resolve(
        input.config.code.targetOS.dylibFileName('wgpu_native'),
      ),
    );
    await destinationFile.parent.create(recursive: true);
    await sourceFile.copy(destinationFile.path);
    return destinationFile.uri;
  }

  @override
  List<Uri> get dependencies => [_resolveLocalFile().uri];
}

/// Builds `wgpu-native` from a local Rust checkout using `cargo`.
final class SourceMode extends BuildMode {
  /// User-provided path to the `wgpu-native` git checkout.
  final Uri? checkoutPath;

  /// Creates a [SourceMode] for [input] and [checkoutPath].
  const SourceMode(super.input, this.checkoutPath);

  @override
  Future<Uri?> build() async {
    final path = checkoutPath;
    if (path == null) {
      throw const FormatException(
        'Specify `checkoutPath` in `hooks.user_defines.gpuarray` '
        '(or `LOCAL_GPUARRAY_CHECKOUT`) to build wgpu-native from source.',
      );
    }
    final directory = Directory.fromUri(path);
    final result = await Process.run('cargo', [
      'build',
      '--release',
    ], workingDirectory: directory.path);
    if (result.exitCode != 0) {
      throw StateError('cargo build failed for wgpu-native:\n${result.stderr}');
    }
    final libraryName = input.config.code.targetOS.dylibFileName('wgpu_native');
    final builtFile = File.fromUri(
      directory.uri.resolve('target/release/$libraryName'),
    );
    if (!builtFile.existsSync()) {
      throw FileSystemException('Built wgpu-native not found', builtFile.path);
    }
    return builtFile.uri;
  }

  @override
  List<Uri> get dependencies =>
      checkoutPath != null ? [checkoutPath!.resolve('Cargo.lock')] : const [];
}

Future<Uint8List> _downloadWithRedirects(Uri url) async {
  final client = HttpClient();
  try {
    var currentUrl = url;
    for (var redirectCount = 0; redirectCount < 5; redirectCount++) {
      final request = await client.getUrl(currentUrl);
      final response = await request.close();
      final location = response.headers.value(HttpHeaders.locationHeader);
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          location != null) {
        currentUrl = currentUrl.resolve(location);
        continue;
      }
      if (response.statusCode != 200) {
        throw HttpException(
          'Failed to download wgpu-native archive from $currentUrl (HTTP ${response.statusCode})',
        );
      }
      final builder = BytesBuilder(copy: false);
      await for (final chunk in response) {
        builder.add(chunk);
      }
      return builder.takeBytes();
    }
    throw HttpException('Too many redirects while downloading $url');
  } finally {
    client.close();
  }
}
