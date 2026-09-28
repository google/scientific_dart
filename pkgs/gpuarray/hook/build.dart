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

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final BuildOptions buildOptions;
    try {
      buildOptions = BuildOptions.fromDefines(input.userDefines);
    } catch (e) {
      throw ArgumentError(BuildOptions.usageError(e));
    }
    print('gpuarray build options: $buildOptions');

    final buildMode = switch (buildOptions.buildMode) {
      BuildModeEnum.fetch => FetchMode(input),
      BuildModeEnum.local => LocalMode(input, buildOptions.localPath),
      BuildModeEnum.source => SourceMode(input, buildOptions.checkoutPath),
    };

    final builtLibrary = await buildMode.build();
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

sealed class BuildMode {
  final BuildInput input;

  const BuildMode(this.input);

  List<Uri> get dependencies;

  Future<Uri?> build();
}

final class FetchMode extends BuildMode {
  FetchMode(super.input);

  @override
  Future<Uri?> build() async {
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final asset = fileHashes[(os, arch)];

    if (asset == null) {
      print(
        'Warning: No prebuilt wgpu-native release configuration for $os $arch. '
        'Hardware acceleration will fall back to CPU simulation mode.',
      );
      return null;
    }

    final extractDir = Directory.fromUri(
      input.outputDirectoryShared.resolve(
        'wgpu-native-$wgpuVersion/${os.name}-${arch.name}/',
      ),
    );
    if (!extractDir.existsSync()) {
      extractDir.createSync(recursive: true);
    }

    final libFile = File(extractDir.uri.resolve(asset.libName).toFilePath());
    if (!libFile.existsSync()) {
      final downloadUrl = Uri.parse('$wgpuBaseUrl/${asset.zipName}');
      print('Downloading wgpu-native release from $downloadUrl...');
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
          final outFile = File(extractDir.uri.resolve(baseName).toFilePath());
          outFile.writeAsBytesSync(file.content as List<int>, flush: true);
        }
      }
    }

    return libFile.existsSync() ? libFile.uri : null;
  }

  @override
  List<Uri> get dependencies => const [];
}

final class LocalMode extends BuildMode {
  final Uri? localPath;

  LocalMode(super.input, this.localPath);

  File _resolveLocalFile() {
    if (localPath == null) {
      throw ArgumentError(
        '`localPath` is not set in `hooks.user_defines.gpuarray` '
        '(or `LOCAL_GPUARRAY_BINARY` environment variable).',
      );
    }
    final file = File(localPath!.toFilePath(windows: Platform.isWindows));
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
    final src = _resolveLocalFile();
    final dst = File.fromUri(
      input.outputDirectory.resolve(
        input.config.code.targetOS.dylibFileName('wgpu_native'),
      ),
    );
    await dst.parent.create(recursive: true);
    await src.copy(dst.path);
    return dst.uri;
  }

  @override
  List<Uri> get dependencies => [_resolveLocalFile().uri];
}

final class SourceMode extends BuildMode {
  final Uri? checkoutPath;

  SourceMode(super.input, this.checkoutPath);

  @override
  Future<Uri?> build() async {
    if (checkoutPath == null) {
      throw ArgumentError(
        'Specify `checkoutPath` in `hooks.user_defines.gpuarray` '
        '(or `LOCAL_GPUARRAY_CHECKOUT`) to build wgpu-native from source.',
      );
    }
    final dir = Directory.fromUri(checkoutPath!);
    final res = await Process.run('cargo', [
      'build',
      '--release',
    ], workingDirectory: dir.path);
    if (res.exitCode != 0) {
      throw StateError('cargo build failed for wgpu-native:\n${res.stderr}');
    }
    final libName = input.config.code.targetOS.dylibFileName('wgpu_native');
    final built = File.fromUri(dir.uri.resolve('target/release/$libName'));
    if (!built.existsSync()) {
      throw FileSystemException('Built wgpu-native not found', built.path);
    }
    return built.uri;
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
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.value(HttpHeaders.locationHeader) != null) {
        currentUrl = currentUrl.resolve(
          response.headers.value(HttpHeaders.locationHeader)!,
        );
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
