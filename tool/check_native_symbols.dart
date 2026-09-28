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

// Verifies that every `@Native` / `@ffi.Native` symbol declared in each
// package's FFI bindings file is exported by the corresponding compiled native
// shared library, and that the embedded `<PKG>_SOURCE_HASH=<sha256>` marker
// matches `computeNativeSourceHash`.

import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:code_assets/code_assets.dart';
import 'package:ndarray/src/hook_helpers/hashes.dart' as ndarray_hashes;
import 'package:openblas/src/hook_helpers/hashes.dart' as openblas_hashes;
import 'package:pocketfft/src/hook_helpers/hashes.dart' as pocketfft_hashes;

const _defaultPackages = ['pocketfft', 'openblas', 'ndarray'];

Future<void> main(List<String> args) async {
  final parser = ArgParser()
    ..addOption(
      'os',
      abbr: 'o',
      allowed: OS.values.map((o) => o.name),
      defaultsTo: OS.current.name,
      help: 'Target operating system of the built artifacts.',
    )
    ..addOption(
      'architecture',
      abbr: 'a',
      allowed: Architecture.values.map((a) => a.name),
      defaultsTo: Architecture.current.name,
      help: 'Target CPU architecture of the built artifacts.',
    )
    ..addMultiOption(
      'package',
      abbr: 'p',
      allowed: _defaultPackages,
      defaultsTo: _defaultPackages,
      help: 'Packages whose built native artifacts should be checked.',
    )
    ..addOption(
      'artifact-dir',
      abbr: 'd',
      defaultsTo: 'dist',
      help: 'Directory containing packaged release artifacts.',
    )
    ..addFlag(
      'verify-source-hash',
      defaultsTo: true,
      help:
          'Verify that the embedded <PKG>_SOURCE_HASH=<sha256> marker matches current hook/ sources.',
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
  } on FormatException catch (e) {
    stderr.writeln('Error: ${e.message}\n\n${parser.usage}');
    exitCode = 2;
    return;
  }

  if (parsed.flag('help')) {
    stdout.writeln('Usage: dart tool/check_native_symbols.dart [options]\n');
    stdout.writeln(parser.usage);
    return;
  }

  final targetOS = OS.fromString(parsed.option('os')!);
  final targetArch = Architecture.fromString(parsed.option('architecture')!);
  final packages = parsed.multiOption('package');
  final artifactDir = Directory(parsed.option('artifact-dir')!);
  final verifySourceHash = parsed.flag('verify-source-hash');

  var hasFailure = false;

  for (final pkg in packages) {
    try {
      await verifyPackageArtifacts(
        packageName: pkg,
        targetOS: targetOS,
        targetArch: targetArch,
        artifactDir: artifactDir,
        workspaceRoot: Directory.current.uri,
        verifySourceHash: verifySourceHash,
      );
    } catch (e) {
      hasFailure = true;
      stderr.writeln('[FAIL] package:$pkg: $e');
    }
  }

  if (hasFailure) {
    exitCode = 1;
  }
}

/// Verifies all `@Native` / `@ffi.Native` symbols (and optional embedded
/// source hash markers) for [packageName]'s built artifacts in [artifactDir].
Future<void> verifyPackageArtifacts({
  required String packageName,
  required OS targetOS,
  required Architecture targetArch,
  required Directory artifactDir,
  required Uri workspaceRoot,
  bool verifySourceHash = true,
}) async {
  final pkgRoot = workspaceRoot.resolve('pkgs/$packageName/');

  switch (packageName) {
    case 'ndarray':
      final artifactName = ndarray_hashes.ndarrayArtifactName(
        targetOS,
        targetArch,
      );
      final libFile = _resolveLibraryFile(
        artifactDir: artifactDir,
        artifactName: artifactName,
        canonicalName: _canonicalName('libndarray', targetOS),
      );
      final expectedSymbols = <String>{
        ...extractDeclaredNativeSymbols(
          File.fromUri(pkgRoot.resolve('lib/src/ndarray_bindings.dart')),
        ),
        ...extractDeclaredNativeSymbols(
          File.fromUri(
            pkgRoot.resolve('lib/src/ndarray_extensions_bindings.dart'),
          ),
        ),
      };
      await verifyLibraryExportsAllSymbols(
        libraryFile: libFile,
        targetOS: targetOS,
        expectedSymbols: expectedSymbols,
      );
      if (verifySourceHash) {
        final expectedHash = ndarray_hashes.computeNativeSourceHash(pkgRoot);
        final embedded = ndarray_hashes.extractEmbeddedSourceHash(
          await libFile.readAsBytes(),
        );
        if (embedded != expectedHash) {
          throw StateError(
            '${libFile.path} embedded source hash ($embedded) does not match '
            'current hook/ source hash ($expectedHash).',
          );
        }
      }
      stdout.writeln(
        '[OK]   package:ndarray (${libFile.uri.pathSegments.last}): '
        'verified ${expectedSymbols.length} exported @Native symbols'
        '${verifySourceHash ? ' + embedded source hash' : ''}.',
      );

    case 'pocketfft':
      final artifactName = pocketfft_hashes.pocketfftArtifactName(
        targetOS,
        targetArch,
      );
      final libFile = _resolveLibraryFile(
        artifactDir: artifactDir,
        artifactName: artifactName,
        canonicalName: _canonicalName('libpocketfft', targetOS),
      );
      final allBindings = extractDeclaredNativeSymbols(
        File.fromUri(pkgRoot.resolve('lib/src/pocketfft_bindings.dart')),
      );
      final expectedSymbols = allBindings
          .where((s) => s.startsWith('kiss_fft'))
          .toSet();
      await verifyLibraryExportsAllSymbols(
        libraryFile: libFile,
        targetOS: targetOS,
        expectedSymbols: expectedSymbols,
      );
      if (verifySourceHash) {
        final expectedHash = pocketfft_hashes.computeNativeSourceHash(pkgRoot);
        final embedded = pocketfft_hashes.extractEmbeddedSourceHash(
          await libFile.readAsBytes(),
        );
        if (embedded != expectedHash) {
          throw StateError(
            '${libFile.path} embedded source hash ($embedded) does not match '
            'current hook/ source hash ($expectedHash).',
          );
        }
      }
      stdout.writeln(
        '[OK]   package:pocketfft (${libFile.uri.pathSegments.last}): '
        'verified ${expectedSymbols.length} exported @Native symbols'
        '${verifySourceHash ? ' + embedded source hash' : ''}.',
      );

    case 'openblas':
      final extArtifact = openblas_hashes.openblasArtifactName(
        targetOS,
        targetArch,
        'openblas_extensions',
      );
      final extFile = _resolveLibraryFile(
        artifactDir: artifactDir,
        artifactName: extArtifact,
        canonicalName: _canonicalName('libopenblas_extensions', targetOS),
      );
      final expectedExtSymbols = extractDeclaredNativeSymbols(
        File.fromUri(
          pkgRoot.resolve('lib/src/openblas_extensions_bindings.dart'),
        ),
      );
      await verifyLibraryExportsAllSymbols(
        libraryFile: extFile,
        targetOS: targetOS,
        expectedSymbols: expectedExtSymbols,
      );
      if (verifySourceHash) {
        final expectedHash = openblas_hashes.computeNativeSourceHash(pkgRoot);
        final embedded = openblas_hashes.extractEmbeddedSourceHash(
          await extFile.readAsBytes(),
        );
        if (embedded != expectedHash) {
          throw StateError(
            '${extFile.path} embedded source hash ($embedded) does not match '
            'current hook/ source hash ($expectedHash).',
          );
        }
      }

      final openblasArtifact = openblas_hashes.openblasArtifactName(
        targetOS,
        targetArch,
        'openblas',
      );
      final openblasFile = _resolveLibraryFile(
        artifactDir: artifactDir,
        artifactName: openblasArtifact,
        canonicalName: _canonicalName('libopenblas', targetOS),
      );
      final allOpenblasBindings = extractDeclaredNativeSymbols(
        File.fromUri(pkgRoot.resolve('lib/src/openblas_bindings.dart')),
      );
      // On macOS, libopenblas.dylib re-exports the system Accelerate framework
      // for cblas_*/LAPACKE_* symbols and defines the openblas_* thread/config
      // symbols directly in the stub library.
      final expectedOpenblasSymbols =
          (targetOS == OS.macOS || targetOS == OS.iOS)
          ? <String>{
              'openblas_get_num_threads',
              'openblas_set_num_threads',
              'openblas_get_config',
            }
          : allOpenblasBindings;
      await verifyLibraryExportsAllSymbols(
        libraryFile: openblasFile,
        targetOS: targetOS,
        expectedSymbols: expectedOpenblasSymbols,
      );
      stdout.writeln(
        '[OK]   package:openblas (${openblasFile.uri.pathSegments.last} + '
        '${extFile.uri.pathSegments.last}): verified '
        '${expectedOpenblasSymbols.length + expectedExtSymbols.length} '
        'exported @Native symbols'
        '${verifySourceHash ? ' + embedded source hash' : ''}.',
      );

    default:
      throw ArgumentError.value(
        packageName,
        'packageName',
        'Must be one of $_defaultPackages.',
      );
  }
}

File _resolveLibraryFile({
  required Directory artifactDir,
  required String artifactName,
  required String canonicalName,
}) {
  final artifactCandidate = File.fromUri(artifactDir.uri.resolve(artifactName));
  if (artifactCandidate.existsSync()) return artifactCandidate;
  final canonicalCandidate = File.fromUri(
    artifactDir.uri.resolve(canonicalName),
  );
  if (canonicalCandidate.existsSync()) return canonicalCandidate;
  throw FileSystemException(
    'Built native library not found ($artifactName or $canonicalName).',
    artifactDir.path,
  );
}

String _canonicalName(String stem, OS os) {
  final ext = switch (os) {
    OS.windows => 'dll',
    OS.macOS || OS.iOS => 'dylib',
    _ => 'so',
  };
  return '$stem.$ext';
}

/// Extracts all `external` function names annotated with `@Native` or
/// `@ffi.Native` in [bindingsFile].
Set<String> extractDeclaredNativeSymbols(File bindingsFile) {
  if (!bindingsFile.existsSync()) {
    throw FileSystemException(
      'Bindings file does not exist.',
      bindingsFile.path,
    );
  }
  final content = bindingsFile.readAsStringSync();
  final pattern = RegExp(
    r'@(?:ffi\.)?Native\b[\s\S]*?external\s+[\w\d_<>.?,\s]+\s+(\w+)\s*\(',
  );
  final symbols = <String>{};
  for (final match in pattern.allMatches(content)) {
    final name = match.group(1);
    if (name != null && name.isNotEmpty) {
      symbols.add(name);
    }
  }
  if (symbols.isEmpty) {
    throw StateError('No @Native symbols found in ${bindingsFile.path}.');
  }
  return symbols;
}

/// Verifies that [libraryFile] exports every symbol in [expectedSymbols].
Future<void> verifyLibraryExportsAllSymbols({
  required File libraryFile,
  required OS targetOS,
  required Set<String> expectedSymbols,
}) async {
  final exported = await extractExportedLibrarySymbols(
    libraryFile: libraryFile,
    targetOS: targetOS,
  );
  final missing = expectedSymbols.difference(exported).toList()..sort();
  if (missing.isNotEmpty) {
    final preview = missing.take(15).join(', ');
    final more = missing.length > 15 ? ' (+${missing.length - 15} more)' : '';
    throw StateError(
      '${libraryFile.path} is missing ${missing.length} of '
      '${expectedSymbols.length} expected @Native symbols: $preview$more',
    );
  }
}

/// Extracts the set of defined exported symbols from [libraryFile].
Future<Set<String>> extractExportedLibrarySymbols({
  required File libraryFile,
  required OS targetOS,
}) async {
  if (!libraryFile.existsSync()) {
    throw FileSystemException('Library file does not exist.', libraryFile.path);
  }

  if (targetOS == OS.windows) {
    final peSymbols = extractPeDllExports(await libraryFile.readAsBytes());
    if (peSymbols.isNotEmpty) {
      return peSymbols;
    }
  }

  final candidates = <(String, List<String>)>[
    if (targetOS == OS.macOS || targetOS == OS.iOS) ...[
      ('nm', ['-gU', libraryFile.path]),
      ('llvm-nm', ['-gU', libraryFile.path]),
    ] else if (targetOS == OS.windows) ...[
      ('llvm-nm', ['-D', '--defined-only', libraryFile.path]),
      (
        r'C:\Program Files\LLVM\bin\llvm-nm.exe',
        ['-D', '--defined-only', libraryFile.path],
      ),
      ('objdump', ['-p', libraryFile.path]),
    ] else ...[
      ('nm', ['-D', '--defined-only', libraryFile.path]),
      ('llvm-nm', ['-D', '--defined-only', libraryFile.path]),
      ('readelf', ['-Ws', '--dyn-syms', libraryFile.path]),
    ],
  ];

  for (final (exe, cmdArgs) in candidates) {
    try {
      final res = await Process.run(exe, cmdArgs);
      if (res.exitCode != 0) continue;
      final out = res.stdout as String;
      final parsed = _parseSymbolToolOutput(out, targetOS: targetOS);
      if (parsed.isNotEmpty) {
        return parsed;
      }
    } on ProcessException {
      continue;
    }
  }

  throw StateError(
    'Unable to inspect exported symbols of ${libraryFile.path} for $targetOS.',
  );
}

Set<String> _parseSymbolToolOutput(String output, {required OS targetOS}) {
  final symbols = <String>{};
  final isDarwin = targetOS == OS.macOS || targetOS == OS.iOS;
  for (final rawLine in output.split('\n')) {
    final line = rawLine.trim();
    if (line.isEmpty) continue;
    // Standard `nm` format: `<addr>? <type> <name>` (including `i` for GNU IFUNC / target_clones)
    final nmMatch = RegExp(
      r'^(?:[0-9a-fA-F]+\s+)?([A-TV-ZatvwIi])\s+(\S+)$',
    ).firstMatch(line);
    if (nmMatch != null) {
      var name = nmMatch.group(2)!;
      final atIdx = name.indexOf('@');
      if (atIdx > 0) name = name.substring(0, atIdx);
      if (isDarwin && name.startsWith('_')) {
        name = name.substring(1);
      }
      symbols.add(name);
      continue;
    }
    // `readelf -Ws` format: `Num: Value Size Type Bind Vis Ndx Name`
    final readelfMatch = RegExp(
      r'^\d+:\s+[0-9a-fA-F]+\s+\d+\s+\S+\s+(?:GLOBAL|WEAK)\s+\S+\s+(?!UND)\S+\s+(\S+)$',
    ).firstMatch(line);
    if (readelfMatch != null) {
      var name = readelfMatch.group(1)!;
      final atIdx = name.indexOf('@');
      if (atIdx > 0) name = name.substring(0, atIdx);
      symbols.add(name);
    }
  }
  return symbols;
}

/// Parses the PE Export Directory Table directly from the bytes of a Windows
/// `.dll` binary.
Set<String> extractPeDllExports(Uint8List bytes) {
  if (bytes.length < 0x40) return const <String>{};
  final data = ByteData.sublistView(bytes);
  // Check 'MZ' magic
  if (data.getUint16(0, Endian.little) != 0x5A4D) return const <String>{};
  final peOffset = data.getUint32(0x3C, Endian.little);
  if (peOffset + 24 > bytes.length) return const <String>{};
  // Check 'PE\0\0' signature
  if (data.getUint32(peOffset, Endian.little) != 0x00004550) {
    return const <String>{};
  }
  final numSections = data.getUint16(peOffset + 6, Endian.little);
  final sizeOfOptionalHeader = data.getUint16(peOffset + 20, Endian.little);
  final optionalHeaderOffset = peOffset + 24;
  if (optionalHeaderOffset + sizeOfOptionalHeader > bytes.length) {
    return const <String>{};
  }
  final magic = data.getUint16(optionalHeaderOffset, Endian.little);
  // PE32 (0x10b) has export data directory at offset 96; PE32+ (0x20b) at 112.
  final dataDirOffset = switch (magic) {
    0x10b => optionalHeaderOffset + 96,
    0x20b => optionalHeaderOffset + 112,
    _ => -1,
  };
  if (dataDirOffset < 0 || dataDirOffset + 8 > bytes.length) {
    return const <String>{};
  }
  final exportRva = data.getUint32(dataDirOffset, Endian.little);
  final exportSize = data.getUint32(dataDirOffset + 4, Endian.little);
  if (exportRva == 0 || exportSize == 0) return const <String>{};

  final sectionTableOffset = optionalHeaderOffset + sizeOfOptionalHeader;
  int? rvaToOffset(int rva) {
    for (var i = 0; i < numSections; i++) {
      final sec = sectionTableOffset + i * 40;
      if (sec + 40 > bytes.length) return null;
      final virtualSize = data.getUint32(sec + 8, Endian.little);
      final virtualAddress = data.getUint32(sec + 12, Endian.little);
      final sizeOfRawData = data.getUint32(sec + 16, Endian.little);
      final pointerToRawData = data.getUint32(sec + 20, Endian.little);
      final span = virtualSize > sizeOfRawData ? virtualSize : sizeOfRawData;
      if (rva >= virtualAddress && rva < virtualAddress + span) {
        return pointerToRawData + (rva - virtualAddress);
      }
    }
    return null;
  }

  final exportDirOffset = rvaToOffset(exportRva);
  if (exportDirOffset == null || exportDirOffset + 40 > bytes.length) {
    return const <String>{};
  }
  final numberOfNames = data.getUint32(exportDirOffset + 24, Endian.little);
  final addressOfNamesRva = data.getUint32(exportDirOffset + 32, Endian.little);
  final namesArrayOffset = rvaToOffset(addressOfNamesRva);
  if (namesArrayOffset == null ||
      namesArrayOffset + numberOfNames * 4 > bytes.length) {
    return const <String>{};
  }

  final exports = <String>{};
  for (var i = 0; i < numberOfNames; i++) {
    final nameRva = data.getUint32(namesArrayOffset + i * 4, Endian.little);
    final nameOffset = rvaToOffset(nameRva);
    if (nameOffset == null || nameOffset >= bytes.length) continue;
    var end = nameOffset;
    while (end < bytes.length && bytes[end] != 0) {
      end++;
    }
    if (end > nameOffset) {
      exports.add(String.fromCharCodes(bytes, nameOffset, end));
    }
  }
  return exports;
}
