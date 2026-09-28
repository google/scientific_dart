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

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:symbolic_dart/src/hook_helpers/build_options.dart';
import 'package:symbolic_dart/src/hook_helpers/hashes.dart';

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
    print('symbolic_dart build options: $buildOptions');

    final buildMode = switch (buildOptions.buildMode) {
      BuildModeEnum.fetch => FetchMode(input),
      BuildModeEnum.local => LocalMode(input, buildOptions.localPath),
      BuildModeEnum.source => SourceMode(input),
    };

    final (:flintUri, :symengineUri) = await buildMode.build();

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'flint',
        linkMode: DynamicLoadingBundled(),
        file: flintUri,
      ),
    );
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'symengine',
        linkMode: DynamicLoadingBundled(),
        file: symengineUri,
      ),
    );
    output.dependencies.addAll(buildMode.dependencies);
    output.dependencies.add(input.packageRoot.resolve('pubspec.yaml'));
  });
}

String _canonicalFlintName(OS os) {
  final prefix = os == OS.windows ? '' : 'lib';
  final ext = os == OS.windows ? '.dll' : (os == OS.macOS ? '.dylib' : '.so');
  return os == OS.windows ? '${prefix}flint$ext' : '${prefix}flint$ext.19';
}

String _canonicalSymengineName(OS os) {
  final prefix = os == OS.windows ? '' : 'lib';
  final ext = os == OS.windows ? '.dll' : (os == OS.macOS ? '.dylib' : '.so');
  return '${prefix}symengine$ext';
}

sealed class BuildMode {
  final BuildInput input;

  const BuildMode(this.input);

  List<Uri> get dependencies;

  Future<({Uri flintUri, Uri symengineUri})> build();
}

final class FetchMode extends BuildMode {
  FetchMode(super.input);

  @override
  Future<({Uri flintUri, Uri symengineUri})> build() async {
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;

    final flintArtifact = symbolicDartArtifactName(os, arch, 'flint');
    final symArtifact = symbolicDartArtifactName(os, arch, 'symengine');
    final expectedFlintHash = fileHashes[(os, arch, 'flint')];
    final expectedSymHash = fileHashes[(os, arch, 'symengine')];

    if (expectedFlintHash == null ||
        expectedFlintHash.startsWith('00000000') ||
        expectedSymHash == null ||
        expectedSymHash.startsWith('00000000')) {
      throw StateError(
        'No prebuilt symbolic_dart binary hashes are pinned for ($os, $arch) in release $version.\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` or `buildMode: local`.')}',
      );
    }

    final sharedDir = input.outputDirectoryShared.resolve(
      'symbolic_dart-$version/${os.name}-${arch.name}/',
    );
    final flintUri = await _fetchOrUseCached(
      cachedFile: File.fromUri(sharedDir.resolve(_canonicalFlintName(os))),
      artifactName: flintArtifact,
      expectedHash: expectedFlintHash,
    );
    final symengineUri = await _fetchOrUseCached(
      cachedFile: File.fromUri(sharedDir.resolve(_canonicalSymengineName(os))),
      artifactName: symArtifact,
      expectedHash: expectedSymHash,
    );

    return (flintUri: flintUri, symengineUri: symengineUri);
  }

  Future<Uri> _fetchOrUseCached({
    required File cachedFile,
    required String artifactName,
    required String expectedHash,
  }) async {
    if (await cachedFile.exists()) {
      final cachedHash = sha256
          .convert(await cachedFile.readAsBytes())
          .toString();
      if (cachedHash == expectedHash) {
        return cachedFile.uri;
      }
    }
    final remoteUri = Uri.parse(
      'https://github.com/$repository/releases/download/$version/$artifactName',
    );
    final bytes = await _downloadBytesWithRedirects(remoteUri);
    final actualHash = sha256.convert(bytes).toString();
    if (actualHash != expectedHash) {
      throw StateError(
        'SHA-256 mismatch for prebuilt symbolic_dart artifact at $remoteUri:\n'
        'Expected: $expectedHash\n'
        'Actual:   $actualHash',
      );
    }
    await cachedFile.parent.create(recursive: true);
    await cachedFile.writeAsBytes(bytes, flush: true);
    return cachedFile.uri;
  }

  @override
  List<Uri> get dependencies => const [];
}

final class LocalMode extends BuildMode {
  final Uri? localPath;

  LocalMode(super.input, this.localPath);

  (File, File) _resolveLocalFiles() {
    if (localPath == null) {
      throw ArgumentError(
        '`localPath` is not set in `hooks.user_defines.symbolic_dart` '
        '(or `LOCAL_SYMBOLIC_DART_BINARY` environment variable).',
      );
    }
    final os = input.config.code.targetOS;
    final dirUri = Directory(
      localPath!.toFilePath(windows: Platform.isWindows),
    ).uri;
    final flint = File.fromUri(dirUri.resolve(_canonicalFlintName(os)));
    final sym = File.fromUri(dirUri.resolve(_canonicalSymengineName(os)));
    if (!flint.existsSync() || !sym.existsSync()) {
      throw FileSystemException(
        'Could not find ${_canonicalFlintName(os)} and ${_canonicalSymengineName(os)} in localPath.',
        dirUri.toFilePath(),
      );
    }
    return (flint, sym);
  }

  @override
  Future<({Uri flintUri, Uri symengineUri})> build() async {
    final (srcFlint, srcSym) = _resolveLocalFiles();
    final os = input.config.code.targetOS;
    final dstFlint = File.fromUri(
      input.outputDirectory.resolve(_canonicalFlintName(os)),
    );
    final dstSym = File.fromUri(
      input.outputDirectory.resolve(_canonicalSymengineName(os)),
    );
    await dstFlint.parent.create(recursive: true);
    await srcFlint.copy(dstFlint.path);
    await srcSym.copy(dstSym.path);
    return (flintUri: dstFlint.uri, symengineUri: dstSym.uri);
  }

  @override
  List<Uri> get dependencies {
    final (srcFlint, srcSym) = _resolveLocalFiles();
    return [srcFlint.uri, srcSym.uri];
  }
}

final class SourceMode extends BuildMode {
  SourceMode(super.input);

  @override
  Future<({Uri flintUri, Uri symengineUri})> build() async {
    final os = input.config.code.targetOS;
    final cCompiler = input.config.code.cCompiler;
    final ccPath = cCompiler?.compiler.toFilePath();

    if (os != OS.linux && os != OS.macOS && os != OS.windows) {
      throw UnimplementedError(
        'symbolic_dart native build supports Linux, macOS, and Windows.',
      );
    }

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }

    final installDir = Directory.fromUri(outputDir.uri.resolve('install'));
    final libPrefix = os == OS.windows ? '' : 'lib';
    final ext = os == OS.windows ? '.dll' : (os == OS.macOS ? '.dylib' : '.so');

    final flintLib = File(
      installDir.uri.resolve('lib/${libPrefix}flint$ext').toFilePath(),
    );
    final symengineLib = File(
      installDir.uri.resolve('lib/${libPrefix}symengine$ext').toFilePath(),
    );
    final mpfrLib = File(
      installDir.uri.resolve('lib/${libPrefix}mpfr$ext').toFilePath(),
    );

    final buildEnv = <String, String>{
      ...Platform.environment,
      'CC': ?ccPath,
      'CXX': ?ccPath,
    };

    final hasSystemMpfrHeader =
        File('/usr/include/mpfr.h').existsSync() ||
        File('/usr/include/x86_64-linux-gnu/mpfr.h').existsSync() ||
        File('/usr/include/aarch64-linux-gnu/mpfr.h').existsSync();

    if (!flintLib.existsSync() || !symengineLib.existsSync()) {
      final srcDir = Directory.fromUri(outputDir.uri.resolve('src'));
      if (!srcDir.existsSync()) {
        srcDir.createSync(recursive: true);
      }

      if (!hasSystemMpfrHeader && !mpfrLib.existsSync()) {
        final mpfrTar = File(
          srcDir.uri.resolve('mpfr-4.2.1.tar.gz').toFilePath(),
        );
        if (!mpfrTar.existsSync()) {
          final res = await Process.run('curl', [
            '-sSL',
            'https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.gz',
            '-o',
            mpfrTar.path,
          ]);
          if (res.exitCode != 0) {
            throw StateError('Failed to download MPFR: ${res.stderr}');
          }
        }

        final mpfrExtract = Directory(
          srcDir.uri.resolve('mpfr-4.2.1').toFilePath(),
        );
        if (!mpfrExtract.existsSync()) {
          await Process.run('tar', [
            '-xzf',
            mpfrTar.path,
          ], workingDirectory: srcDir.path);
        }

        final gmpInc = File('/usr/include/x86_64-linux-gnu/gmp.h').existsSync()
            ? '/usr/include/x86_64-linux-gnu'
            : '/usr/include';
        final confRes = await Process.run(
          './configure',
          [
            '--prefix=${installDir.path}',
            '--with-gmp-include=$gmpInc',
            '--disable-static',
            '--enable-shared',
          ],
          workingDirectory: mpfrExtract.path,
          environment: buildEnv,
        );
        if (confRes.exitCode != 0) {
          throw StateError('MPFR configure failed:\n${confRes.stderr}');
        }

        final makeRes = await Process.run(
          'make',
          ['-j${Platform.numberOfProcessors}', 'install'],
          workingDirectory: mpfrExtract.path,
          environment: buildEnv,
        );
        if (makeRes.exitCode != 0) {
          throw StateError('MPFR make install failed:\n${makeRes.stderr}');
        }
      }

      if (!flintLib.existsSync()) {
        final flintTar = File(
          srcDir.uri.resolve('flint-3.1.3.tar.gz').toFilePath(),
        );
        if (!flintTar.existsSync()) {
          final res = await Process.run('curl', [
            '-sSL',
            'https://github.com/flintlib/flint/releases/download/v3.1.3/flint-3.1.3.tar.gz',
            '-o',
            flintTar.path,
          ]);
          if (res.exitCode != 0) {
            throw StateError('Failed to download FLINT: ${res.stderr}');
          }
        }

        final flintExtract = Directory(
          srcDir.uri.resolve('flint-3.1.3').toFilePath(),
        );
        if (!flintExtract.existsSync()) {
          await Process.run('tar', [
            '-xzf',
            flintTar.path,
          ], workingDirectory: srcDir.path);
        }

        final gmpInc = File('/usr/include/x86_64-linux-gnu/gmp.h').existsSync()
            ? '/usr/include/x86_64-linux-gnu'
            : '/usr/include';
        final confArgs = [
          '--prefix=${installDir.path}',
          '--with-gmp-include=$gmpInc',
          '--disable-static',
          '--enable-shared',
        ];
        if (!hasSystemMpfrHeader) {
          confArgs.add('--with-mpfr=${installDir.path}');
        }

        final confRes = await Process.run(
          './configure',
          confArgs,
          workingDirectory: flintExtract.path,
          environment: buildEnv,
        );
        if (confRes.exitCode != 0) {
          throw StateError('FLINT configure failed:\n${confRes.stderr}');
        }

        final makeRes = await Process.run(
          'make',
          ['-j${Platform.numberOfProcessors}', 'install'],
          workingDirectory: flintExtract.path,
          environment: buildEnv,
        );
        if (makeRes.exitCode != 0) {
          throw StateError('FLINT make install failed:\n${makeRes.stderr}');
        }
      }

      if (!symengineLib.existsSync()) {
        final symTar = File(
          srcDir.uri.resolve('symengine-0.11.2.tar.gz').toFilePath(),
        );
        if (!symTar.existsSync()) {
          final res = await Process.run('curl', [
            '-sSL',
            'https://github.com/symengine/symengine/releases/download/v0.11.2/symengine-0.11.2.tar.gz',
            '-o',
            symTar.path,
          ]);
          if (res.exitCode != 0) {
            throw StateError('Failed to download SymEngine: ${res.stderr}');
          }
        }

        final symExtract = Directory(
          srcDir.uri.resolve('symengine-0.11.2').toFilePath(),
        );
        if (!symExtract.existsSync()) {
          await Process.run('tar', [
            '-xzf',
            symTar.path,
          ], workingDirectory: srcDir.path);
        }

        final symBuild = Directory(
          symExtract.uri.resolve('build').toFilePath(),
        );
        symBuild.createSync(recursive: true);

        final pkgConfigPath = installDir.uri
            .resolve('lib/pkgconfig')
            .toFilePath();
        final gmpInc = File('/usr/include/x86_64-linux-gnu/gmp.h').existsSync()
            ? '/usr/include/x86_64-linux-gnu'
            : '/usr/include';

        final cmakeArgs = [
          '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
          '-DBUILD_SHARED_LIBS=ON',
          '-DBUILD_TESTS=OFF',
          '-DBUILD_BENCHMARKS=OFF',
          '-DWITH_GMP=ON',
          '-DGMP_INCLUDE_DIR=$gmpInc',
          '-DWITH_FLINT=ON',
          '-DFLINT_INCLUDE_DIR=${installDir.uri.resolve('include').toFilePath()}',
          '-DFLINT_LIBRARY=${flintLib.path}',
          '-DCMAKE_INSTALL_PREFIX=${installDir.path}',
          '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
          if (os != OS.windows) '-DCMAKE_INSTALL_RPATH=\$ORIGIN',
          if (ccPath != null) ...[
            '-DCMAKE_C_COMPILER=$ccPath',
            '-DCMAKE_CXX_COMPILER=$ccPath',
          ],
          '..',
        ];

        if (!hasSystemMpfrHeader) {
          cmakeArgs.addAll([
            '-DWITH_MPFR=ON',
            '-DMPFR_INCLUDE_DIR=${installDir.uri.resolve('include').toFilePath()}',
            '-DMPFR_LIBRARY=${mpfrLib.path}',
          ]);
        }

        final cmakeRes = await Process.run(
          'cmake',
          cmakeArgs,
          workingDirectory: symBuild.path,
          environment: {'PKG_CONFIG_PATH': pkgConfigPath, ...buildEnv},
        );
        if (cmakeRes.exitCode != 0) {
          throw StateError('SymEngine CMake failed:\n${cmakeRes.stderr}');
        }

        final makeRes = await Process.run(
          'make',
          ['-j${Platform.numberOfProcessors}', 'install'],
          workingDirectory: symBuild.path,
          environment: buildEnv,
        );
        if (makeRes.exitCode != 0) {
          throw StateError('SymEngine make failed:\n${makeRes.stderr}');
        }
      }
    }

    final File assetFlintFile;
    if (os == OS.windows) {
      assetFlintFile = flintLib;
    } else {
      final flint19Lib = File(
        installDir.uri.resolve('lib/${libPrefix}flint$ext.19').toFilePath(),
      );
      if (!flint19Lib.existsSync()) {
        flintLib.copySync(flint19Lib.path);
      }
      assetFlintFile = flint19Lib;
    }

    return (flintUri: assetFlintFile.uri, symengineUri: symengineLib.uri);
  }

  @override
  List<Uri> get dependencies => const [];
}

Future<Uint8List> _downloadBytesWithRedirects(Uri url) async {
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
          'Failed to download $currentUrl (HTTP ${response.statusCode})',
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
