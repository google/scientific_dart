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

// Downloads precompiled binary artifacts for a GitHub Release (or reads them
// from a local directory), verifies their SLSA v1.0 build provenance via
// `gh attestation verify`, computes SHA-256 digests, and updates
// `pkgs/<pkg>/lib/src/hook_helpers/hashes.dart`.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';

const _supportedTargets = <(OS, Architecture)>[
  (OS.linux, Architecture.x64),
  (OS.linux, Architecture.arm64),
  (OS.macOS, Architecture.arm64),
  (OS.macOS, Architecture.x64),
  (OS.windows, Architecture.x64),
];

Future<void> main(List<String> args) async {
  final parser = ArgParser()
    ..addOption(
      'repo',
      abbr: 'r',
      defaultsTo: 'google/scientific_dart',
      help:
          'GitHub repository (<owner>/<name>) hosting releases and attestations.',
    )
    ..addOption(
      'local-dir',
      abbr: 'l',
      help:
          'Optional local directory containing built artifacts (instead of downloading from GitHub).',
    )
    ..addFlag(
      'skip-provenance',
      negatable: false,
      help:
          'Skip `gh attestation verify` check (only use for local dry runs before release).',
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

  if (parsed.flag('help') || parsed.rest.isEmpty) {
    stdout.writeln(
      'Usage: dart tool/regenerate_hashes.dart <release-tag> [options]\n',
    );
    stdout.writeln(parser.usage);
    exit(parsed.flag('help') ? 0 : 64);
  }

  final version = parsed.rest.first;
  final repo = parsed.option('repo')!;
  final localDirOption = parsed.option('local-dir');
  final skipProvenance = parsed.flag('skip-provenance');
  final localDir = localDirOption != null ? Directory(localDirOption) : null;

  final tempDir = await Directory.systemTemp.createTemp(
    'scientific_dart_hashes_',
  );
  try {
    // 1. Update pocketfft hashes
    await _regenerateSingleAssetHashes(
      packageName: 'pocketfft',
      assetStem: 'pocketfft',
      helperFuncName: 'pocketfftArtifactName',
      version: version,
      repo: repo,
      localDir: localDir,
      tempDir: tempDir,
      skipProvenance: skipProvenance,
    );

    // 2. Update ndarray hashes
    await _regenerateSingleAssetHashes(
      packageName: 'ndarray',
      assetStem: 'ndarray',
      helperFuncName: 'ndarrayArtifactName',
      version: version,
      repo: repo,
      localDir: localDir,
      tempDir: tempDir,
      skipProvenance: skipProvenance,
    );

    // 3. Update openblas hashes (openblas + openblas_extensions)
    await _regenerateMultiAssetHashes(
      packageName: 'openblas',
      assetStems: const ['openblas', 'openblas_extensions'],
      helperFuncName: 'openblasArtifactName',
      version: version,
      repo: repo,
      localDir: localDir,
      tempDir: tempDir,
      skipProvenance: skipProvenance,
    );
  } finally {
    try {
      await tempDir.delete(recursive: true);
    } catch (_) {}
  }
}

Future<void> _regenerateSingleAssetHashes({
  required String packageName,
  required String assetStem,
  required String helperFuncName,
  required String version,
  required String repo,
  required Directory? localDir,
  required Directory tempDir,
  required bool skipProvenance,
}) async {
  stdout.writeln('=== Updating hashes for package:$packageName ($version) ===');
  final hashes = <(OS, Architecture), String>{};

  for (final (os, arch) in _supportedTargets) {
    final ext = _extForOS(os);
    final artifactName = '$assetStem-${os.name}-${arch.name}.$ext';
    final file = await _obtainArtifact(
      artifactName: artifactName,
      version: version,
      repo: repo,
      localDir: localDir,
      tempDir: tempDir,
    );

    if (file == null) {
      stdout.writeln('  [skip] $artifactName not found');
      hashes[(os, arch)] = '0' * 64;
      continue;
    }

    if (!skipProvenance) {
      await _verifyProvenance(file: file, repo: repo);
    }

    final digest = sha256.convert(await file.readAsBytes()).toString();
    hashes[(os, arch)] = digest;
    stdout.writeln('  [ok]   $artifactName -> $digest');
  }

  final sourceHash = await _computeNativeSourceHashForRef(
    packageName: packageName,
    gitRef: localDir == null ? version : null,
  );
  stdout.writeln('  [src]  nativeSourceHash -> $sourceHash');

  final markerPrefix = '${packageName.toUpperCase()}_SOURCE_HASH=';

  for (final (os, arch) in _supportedTargets) {
    final ext = _extForOS(os);
    final artifactName = '$assetStem-${os.name}-${arch.name}.$ext';
    final file = await _obtainArtifact(
      artifactName: artifactName,
      version: version,
      repo: repo,
      localDir: localDir,
      tempDir: tempDir,
    );
    if (file == null) continue;
    final bytes = await file.readAsBytes();
    final embedded = _extractEmbeddedHash(bytes, markerPrefix);
    if (embedded != null && embedded != sourceHash) {
      throw StateError(
        'Artifact $artifactName embeds $markerPrefix$embedded, which does not '
        'match computed hook/ source hash $sourceHash for $packageName.',
      );
    }
  }

  final entriesCode = hashes.entries
      .map(
        (e) =>
            '  (${_osConstName(e.key.$1)}, Architecture.${e.key.$2.name}):\n'
            "      '${e.value}',",
      )
      .join('\n');

  final targetFile = File('pkgs/$packageName/lib/src/hook_helpers/hashes.dart');
  await targetFile.writeAsString('''
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

// THIS FILE IS AUTOGENERATED BY `tool/regenerate_hashes.dart`. TO UPDATE, RUN:
//
//   dart tool/regenerate_hashes.dart <github release tag>
//

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';

/// GitHub repository hosting prebuilt release artifacts and SLSA provenance attestations.
const repository = '$repo';

/// Release tag for prebuilt `$packageName` binaries.
const version = '$version';

/// Combined SHA-256 digest of `hook/` native source files at [version].
const nativeSourceHash = '$sourceHash';

/// ASCII marker prefix embedded in compiled `$packageName` binaries.
const sourceHashMarkerPrefix = '$markerPrefix';

/// Extracts the 64-character lowercase hex SHA-256 digest embedded after
/// [sourceHashMarkerPrefix] in [bytes], or returns `null` if no marker is
/// present.
String? extractEmbeddedSourceHash(Uint8List bytes) {
  final prefix = sourceHashMarkerPrefix.codeUnits;
  final prefixLen = prefix.length;
  const hashLen = 64;
  final limit = bytes.length - (prefixLen + hashLen);
  outer:
  for (var i = 0; i <= limit; i++) {
    for (var j = 0; j < prefixLen; j++) {
      if (bytes[i + j] != prefix[j]) continue outer;
    }
    final start = i + prefixLen;
    for (var k = 0; k < hashLen; k++) {
      final b = bytes[start + k];
      final isDigit = b >= 0x30 && b <= 0x39;
      final isHexLower = b >= 0x61 && b <= 0x66;
      if (!isDigit && !isHexLower) continue outer;
    }
    return String.fromCharCodes(bytes, start, start + hashLen);
  }
  return null;
}

/// Verifies that [bytes] of a prebuilt `$packageName` artifact correspond to
/// [currentSourceHash].
///
/// It is an error if [currentSourceHash] differs from [nativeSourceHash], if
/// an embedded [sourceHashMarkerPrefix] marker is present and does not match
/// [currentSourceHash], or if the marker is missing in a release after
/// `artifacts-v0.0.2`.
void verifyArtifactSourceHash(
  Uint8List bytes, {
  required String currentSourceHash,
  String releaseVersion = version,
}) {
  if (currentSourceHash != nativeSourceHash) {
    throw StateError(
      'Prebuilt $packageName binary for release \$releaseVersion is out of date with '
      'native sources in hook/ (pinned: \$nativeSourceHash, '
      'current: \$currentSourceHash).',
    );
  }
  final embeddedHash = extractEmbeddedSourceHash(bytes);
  if (embeddedHash == null) {
    if (releaseVersion != 'artifacts-v0.0.2') {
      throw StateError(
        'Prebuilt $packageName binary for release \$releaseVersion is missing '
        'embedded \$sourceHashMarkerPrefix<sha256> marker.',
      );
    }
    return;
  }
  if (embeddedHash != currentSourceHash) {
    throw StateError(
      'Embedded source hash in prebuilt $packageName binary (\$embeddedHash) does '
      'not match hook/ source hash (\$currentSourceHash).',
    );
  }
}

/// Lists the tracked native source files in `hook/` under [packageRoot].
List<File> nativeSourceFiles(Uri packageRoot) {
  final hookDir = Directory.fromUri(packageRoot.resolve('hook/'));
  if (!hookDir.existsSync()) return const [];
  final files = hookDir.listSync().whereType<File>().where((file) {
    final name = file.uri.pathSegments.last;
    return name.endsWith('.dart') ||
        name.endsWith('.c') ||
        name.endsWith('.cpp') ||
        name.endsWith('.h') ||
        name.endsWith('.def');
  }).toList();
  for (final relPath in const [
    'third_party/miniz/miniz.c',
    'third_party/miniz/miniz.h',
  ]) {
    final extra = File.fromUri(packageRoot.resolve(relPath));
    if (extra.existsSync()) {
      files.add(extra);
    }
  }
  files.sort(
    (a, b) => a.uri.pathSegments.last.compareTo(b.uri.pathSegments.last),
  );
  return files;
}

/// Computes the combined SHA-256 digest of `hook/` native source files under [packageRoot].
String computeNativeSourceHash(Uri packageRoot) {
  final buffer = StringBuffer();
  for (final file in nativeSourceFiles(packageRoot)) {
    final name = file.uri.pathSegments.last;
    final normalized = file.readAsStringSync().replaceAll('\\r\\n', '\\n');
    final fileDigest = sha256.convert(utf8.encode(normalized)).toString();
    buffer.writeln('\$name:\$fileDigest');
  }
  return sha256.convert(utf8.encode(buffer.toString())).toString();
}

/// Canonical release artifact filename for `(os, arch)`.
String $helperFuncName(OS os, Architecture arch) {
  final ext = switch (os) {
    OS.windows => 'dll',
    OS.macOS || OS.iOS => 'dylib',
    _ => 'so',
  };
  return '$assetStem-\${os.name}-\${arch.name}.\$ext';
}

/// SHA-256 digests for prebuilt `$packageName` binaries indexed by `(OS, Architecture)`.
final fileHashes = <(OS, Architecture), String>{
$entriesCode
};
''');
  stdout.writeln('Wrote ${targetFile.path}');
}

Future<void> _regenerateMultiAssetHashes({
  required String packageName,
  required List<String> assetStems,
  required String helperFuncName,
  required String version,
  required String repo,
  required Directory? localDir,
  required Directory tempDir,
  required bool skipProvenance,
}) async {
  stdout.writeln('=== Updating hashes for package:$packageName ($version) ===');
  final hashes = <(OS, Architecture, String), String>{};

  for (final (os, arch) in _supportedTargets) {
    final ext = _extForOS(os);
    for (final stem in assetStems) {
      final artifactName = '$stem-${os.name}-${arch.name}.$ext';
      final file = await _obtainArtifact(
        artifactName: artifactName,
        version: version,
        repo: repo,
        localDir: localDir,
        tempDir: tempDir,
      );

      if (file == null) {
        stdout.writeln('  [skip] $artifactName not found');
        hashes[(os, arch, stem)] = '0' * 64;
        continue;
      }

      if (!skipProvenance) {
        await _verifyProvenance(file: file, repo: repo);
      }

      final digest = sha256.convert(await file.readAsBytes()).toString();
      hashes[(os, arch, stem)] = digest;
      stdout.writeln('  [ok]   $artifactName -> $digest');
    }
  }

  final sourceHash = await _computeNativeSourceHashForRef(
    packageName: packageName,
    gitRef: localDir == null ? version : null,
  );
  stdout.writeln('  [src]  nativeSourceHash -> $sourceHash');

  final markerPrefix = '${packageName.toUpperCase()}_SOURCE_HASH=';

  for (final (os, arch) in _supportedTargets) {
    final ext = _extForOS(os);
    for (final stem in assetStems) {
      final artifactName = '$stem-${os.name}-${arch.name}.$ext';
      final file = await _obtainArtifact(
        artifactName: artifactName,
        version: version,
        repo: repo,
        localDir: localDir,
        tempDir: tempDir,
      );
      if (file == null) continue;
      final bytes = await file.readAsBytes();
      final embedded = _extractEmbeddedHash(bytes, markerPrefix);
      if (embedded != null && embedded != sourceHash) {
        throw StateError(
          'Artifact $artifactName embeds $markerPrefix$embedded, which does not '
          'match computed hook/ source hash $sourceHash for $packageName.',
        );
      }
    }
  }

  final entriesCode = hashes.entries
      .map(
        (e) =>
            "  (${_osConstName(e.key.$1)}, Architecture.${e.key.$2.name}, '${e.key.$3}'):\n"
            "      '${e.value}',",
      )
      .join('\n');

  final targetFile = File('pkgs/$packageName/lib/src/hook_helpers/hashes.dart');
  await targetFile.writeAsString('''
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

// THIS FILE IS AUTOGENERATED BY `tool/regenerate_hashes.dart`. TO UPDATE, RUN:
//
//   dart tool/regenerate_hashes.dart <github release tag>
//

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';

/// GitHub repository hosting prebuilt release artifacts and SLSA provenance attestations.
const repository = '$repo';

/// Release tag for prebuilt `$packageName` binaries.
const version = '$version';

/// Combined SHA-256 digest of `hook/` native source files at [version].
const nativeSourceHash = '$sourceHash';

/// ASCII marker prefix embedded in compiled `$packageName` extension binaries.
const sourceHashMarkerPrefix = '$markerPrefix';

/// Extracts the 64-character lowercase hex SHA-256 digest embedded after
/// [sourceHashMarkerPrefix] in [bytes], or returns `null` if no marker is
/// present.
String? extractEmbeddedSourceHash(Uint8List bytes) {
  final prefix = sourceHashMarkerPrefix.codeUnits;
  final prefixLen = prefix.length;
  const hashLen = 64;
  final limit = bytes.length - (prefixLen + hashLen);
  outer:
  for (var i = 0; i <= limit; i++) {
    for (var j = 0; j < prefixLen; j++) {
      if (bytes[i + j] != prefix[j]) continue outer;
    }
    final start = i + prefixLen;
    for (var k = 0; k < hashLen; k++) {
      final b = bytes[start + k];
      final isDigit = b >= 0x30 && b <= 0x39;
      final isHexLower = b >= 0x61 && b <= 0x66;
      if (!isDigit && !isHexLower) continue outer;
    }
    return String.fromCharCodes(bytes, start, start + hashLen);
  }
  return null;
}

/// Verifies that [bytes] of a prebuilt `$packageName` artifact correspond to
/// [currentSourceHash].
///
/// It is an error if [currentSourceHash] differs from [nativeSourceHash], if
/// an embedded [sourceHashMarkerPrefix] marker is present and does not match
/// [currentSourceHash], or if the marker is missing in a release after
/// `artifacts-v0.0.2`.
void verifyArtifactSourceHash(
  Uint8List bytes, {
  required String currentSourceHash,
  String releaseVersion = version,
}) {
  if (currentSourceHash != nativeSourceHash) {
    throw StateError(
      'Prebuilt $packageName binary for release \$releaseVersion is out of date with '
      'native sources in hook/ (pinned: \$nativeSourceHash, '
      'current: \$currentSourceHash).',
    );
  }
  final embeddedHash = extractEmbeddedSourceHash(bytes);
  if (embeddedHash == null) {
    if (releaseVersion != 'artifacts-v0.0.2') {
      throw StateError(
        'Prebuilt $packageName binary for release \$releaseVersion is missing '
        'embedded \$sourceHashMarkerPrefix<sha256> marker.',
      );
    }
    return;
  }
  if (embeddedHash != currentSourceHash) {
    throw StateError(
      'Embedded source hash in prebuilt $packageName binary (\$embeddedHash) does '
      'not match hook/ source hash (\$currentSourceHash).',
    );
  }
}

/// Lists the tracked native source files in `hook/` (and `third_party/miniz/`)
/// under [packageRoot].
List<File> nativeSourceFiles(Uri packageRoot) {
  final hookDir = Directory.fromUri(packageRoot.resolve('hook/'));
  if (!hookDir.existsSync()) return const [];
  final files = hookDir.listSync().whereType<File>().where((file) {
    final name = file.uri.pathSegments.last;
    return name.endsWith('.dart') ||
        name.endsWith('.c') ||
        name.endsWith('.cpp') ||
        name.endsWith('.h') ||
        name.endsWith('.def');
  }).toList();
  for (final relPath in const [
    'third_party/miniz/miniz.c',
    'third_party/miniz/miniz.h',
  ]) {
    final file = File.fromUri(packageRoot.resolve(relPath));
    if (file.existsSync()) {
      files.add(file);
    }
  }
  files.sort(
    (a, b) => a.uri.pathSegments.last.compareTo(b.uri.pathSegments.last),
  );
  return files;
}

/// Computes the combined SHA-256 digest of `hook/` native source files under [packageRoot].
String computeNativeSourceHash(Uri packageRoot) {
  final buffer = StringBuffer();
  for (final file in nativeSourceFiles(packageRoot)) {
    final name = file.uri.pathSegments.last;
    final normalized = file.readAsStringSync().replaceAll('\\r\\n', '\\n');
    final fileDigest = sha256.convert(utf8.encode(normalized)).toString();
    buffer.writeln('\$name:\$fileDigest');
  }
  return sha256.convert(utf8.encode(buffer.toString())).toString();
}

/// Canonical release artifact filename for `(os, arch, libraryKind)`.
///
/// [libraryKind] is either `'openblas'` or `'openblas_extensions'`.
String $helperFuncName(OS os, Architecture arch, String libraryKind) {
  final ext = switch (os) {
    OS.windows => 'dll',
    OS.macOS || OS.iOS => 'dylib',
    _ => 'so',
  };
  return '\$libraryKind-\${os.name}-\${arch.name}.\$ext';
}

/// SHA-256 digests for prebuilt `$packageName` binaries indexed by `(OS, Architecture, libraryKind)`.
final fileHashes = <(OS, Architecture, String), String>{
$entriesCode
};
''');
  stdout.writeln('Wrote ${targetFile.path}');
}

String? _extractEmbeddedHash(Uint8List bytes, String markerPrefix) {
  final prefix = markerPrefix.codeUnits;
  final prefixLen = prefix.length;
  const hashLen = 64;
  final limit = bytes.length - (prefixLen + hashLen);
  outer:
  for (var i = 0; i <= limit; i++) {
    for (var j = 0; j < prefixLen; j++) {
      if (bytes[i + j] != prefix[j]) continue outer;
    }
    final start = i + prefixLen;
    for (var k = 0; k < hashLen; k++) {
      final b = bytes[start + k];
      final isDigit = b >= 0x30 && b <= 0x39;
      final isHexLower = b >= 0x61 && b <= 0x66;
      if (!isDigit && !isHexLower) continue outer;
    }
    return String.fromCharCodes(bytes, start, start + hashLen);
  }
  return null;
}

Future<String> _computeNativeSourceHashForRef({
  required String packageName,
  required String? gitRef,
}) async {
  bool isTrackedSource(String name) =>
      name.endsWith('.dart') ||
      name.endsWith('.c') ||
      name.endsWith('.cpp') ||
      name.endsWith('.h') ||
      name.endsWith('.def');

  if (gitRef != null) {
    final lsTree = await Process.run('git', [
      'ls-tree',
      '--name-only',
      gitRef,
      'pkgs/$packageName/hook/',
      if (packageName == 'ndarray') ...[
        'pkgs/$packageName/third_party/miniz/miniz.c',
        'pkgs/$packageName/third_party/miniz/miniz.h',
      ],
    ]);
    if (lsTree.exitCode == 0) {
      final paths = (lsTree.stdout as String)
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty && isTrackedSource(s))
          .toList();
      if (paths.isNotEmpty) {
        paths.sort((a, b) => a.split('/').last.compareTo(b.split('/').last));
        final buffer = StringBuffer();
        for (final path in paths) {
          final name = path.split('/').last;
          final showRes = await Process.run('git', ['show', '$gitRef:$path']);
          if (showRes.exitCode != 0) {
            throw StateError('Failed to read $gitRef:$path via git show');
          }
          final content = showRes.stdout as String;
          final normalized = content.replaceAll('\r\n', '\n');
          final fileDigest = sha256.convert(utf8.encode(normalized)).toString();
          buffer.writeln('$name:$fileDigest');
        }
        return sha256.convert(utf8.encode(buffer.toString())).toString();
      }
    }
  }

  final hookDir = Directory('pkgs/$packageName/hook');
  final files = hookDir
      .listSync()
      .whereType<File>()
      .where((f) => isTrackedSource(f.uri.pathSegments.last))
      .toList();
  for (final relPath in const [
    'third_party/miniz/miniz.c',
    'third_party/miniz/miniz.h',
  ]) {
    final extra = File('pkgs/$packageName/$relPath');
    if (extra.existsSync()) {
      files.add(extra);
    }
  }
  files.sort(
    (a, b) => a.uri.pathSegments.last.compareTo(b.uri.pathSegments.last),
  );
  final buffer = StringBuffer();
  for (final file in files) {
    final name = file.uri.pathSegments.last;
    final normalized = file.readAsStringSync().replaceAll('\r\n', '\n');
    final fileDigest = sha256.convert(utf8.encode(normalized)).toString();
    buffer.writeln('$name:$fileDigest');
  }
  return sha256.convert(utf8.encode(buffer.toString())).toString();
}

String _osConstName(OS os) => switch (os) {
  OS.macOS => 'OS.macOS',
  OS.iOS => 'OS.iOS',
  _ => 'OS.${os.name}',
};

String _extForOS(OS os) => switch (os) {
  OS.windows => 'dll',
  OS.macOS || OS.iOS => 'dylib',
  _ => 'so',
};

Future<File?> _obtainArtifact({
  required String artifactName,
  required String version,
  required String repo,
  required Directory? localDir,
  required Directory tempDir,
}) async {
  if (localDir != null) {
    final candidate = File.fromUri(localDir.uri.resolve(artifactName));
    return candidate.existsSync() ? candidate : null;
  }

  final uri = Uri.parse(
    'https://github.com/$repo/releases/download/$version/$artifactName',
  );
  final bytes = await _downloadWithRedirects(uri);
  if (bytes == null) return null;

  final dst = File.fromUri(tempDir.uri.resolve(artifactName));
  await dst.writeAsBytes(bytes, flush: true);
  return dst;
}

Future<void> _verifyProvenance({
  required File file,
  required String repo,
}) async {
  stdout.writeln(
    '  [provenance] Verifying SLSA attestation for ${file.uri.pathSegments.last}...',
  );
  final res = await Process.run('gh', [
    'attestation',
    'verify',
    file.path,
    '--repo',
    repo,
  ]);
  if (res.exitCode != 0) {
    throw StateError(
      'SLSA provenance verification failed for ${file.path} against $repo:\n'
      'stdout: ${res.stdout}\n'
      'stderr: ${res.stderr}',
    );
  }
}

Future<Uint8List?> _downloadWithRedirects(Uri url) async {
  final client = HttpClient();
  try {
    var currentUri = url;
    for (var redirectCount = 0; redirectCount < 5; redirectCount++) {
      final request = await client.getUrl(currentUri);
      final response = await request.close();
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.value(HttpHeaders.locationHeader) != null) {
        final location = response.headers.value(HttpHeaders.locationHeader)!;
        await response.drain<void>();
        final nextUri = currentUri.resolve(location);
        if (currentUri.scheme == 'https' && nextUri.scheme != 'https') {
          throw HttpException(
            'Refusing HTTPS-to-HTTP redirect downgrade: $nextUri',
          );
        }
        currentUri = nextUri;
        continue;
      }
      if (response.statusCode == 404) {
        await response.drain<void>();
        return null;
      }
      if (response.statusCode != 200) {
        await response.drain<void>();
        throw HttpException(
          'Failed to download $currentUri (HTTP ${response.statusCode})',
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
