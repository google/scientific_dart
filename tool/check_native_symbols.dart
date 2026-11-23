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
// shared library, that the embedded `<PKG>_SOURCE_HASH=<sha256>` marker
// matches `computeNativeSourceHash`, and that Android/iOS binaries satisfy
// platform ABI hygiene invariants (16 KB ELF alignment, static C++ runtime,
// Mach-O LC_BUILD_VERSION platform, @rpath install_name).

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
    ..addOption(
      'ios-sdk',
      allowed: IOSSdk.values.map((s) => s.type),
      help:
          'Target iOS SDK (`iphoneos` or `iphonesimulator`, required for `--os ios`).',
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
  final rawIosSdk = parsed.option('ios-sdk');
  final IOSSdk? targetIosSdk;
  if (targetOS == OS.iOS) {
    if (rawIosSdk == null) {
      stderr.writeln('Error: `--ios-sdk` is required when `--os ios` is set.');
      exitCode = 2;
      return;
    }
    targetIosSdk = IOSSdk.fromString(rawIosSdk);
  } else {
    if (rawIosSdk != null) {
      stderr.writeln(
        'Error: `--ios-sdk` is only valid when `--os ios` is set.',
      );
      exitCode = 2;
      return;
    }
    targetIosSdk = null;
  }
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
        targetIosSdk: targetIosSdk,
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
/// source hash markers and mobile binary hygiene invariants) for
/// [packageName]'s built artifacts in [artifactDir].
Future<void> verifyPackageArtifacts({
  required String packageName,
  required OS targetOS,
  required Architecture targetArch,
  IOSSdk? targetIosSdk,
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
        iosSdk: targetIosSdk,
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
      await verifyPlatformBinaryHygiene(
        libraryFile: libFile,
        targetOS: targetOS,
        targetArch: targetArch,
        targetIosSdk: targetIosSdk,
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
        iosSdk: targetIosSdk,
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
      await verifyPlatformBinaryHygiene(
        libraryFile: libFile,
        targetOS: targetOS,
        targetArch: targetArch,
        targetIosSdk: targetIosSdk,
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
        iosSdk: targetIosSdk,
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
      await verifyPlatformBinaryHygiene(
        libraryFile: extFile,
        targetOS: targetOS,
        targetArch: targetArch,
        targetIosSdk: targetIosSdk,
        expectedAndroidNeeded: targetOS == OS.android ? 'libopenblas.so' : null,
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
        iosSdk: targetIosSdk,
      );
      final openblasFile = _resolveLibraryFile(
        artifactDir: artifactDir,
        artifactName: openblasArtifact,
        canonicalName: _canonicalName('libopenblas', targetOS),
      );
      final expectedOpenblasSymbols = <String>{
        ...extractDeclaredNativeSymbols(
          File.fromUri(pkgRoot.resolve('lib/src/openblas_bindings.dart')),
        ),
        ...extractDeclaredNativeSymbols(
          File.fromUri(pkgRoot.resolve('lib/src/openblas_extensions.dart')),
        ),
      };
      await verifyLibraryExportsAllSymbols(
        libraryFile: openblasFile,
        targetOS: targetOS,
        expectedSymbols: expectedOpenblasSymbols,
      );
      await verifyPlatformBinaryHygiene(
        libraryFile: openblasFile,
        targetOS: targetOS,
        targetArch: targetArch,
        targetIosSdk: targetIosSdk,
        expectedAndroidSoname: targetOS == OS.android ? 'libopenblas.so' : null,
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

/// Verifies platform-specific binary hygiene invariants for Android and iOS
/// artifacts (16 KB ELF alignment, runtime link dependencies, Mach-O platform
/// & install_name).
Future<void> verifyPlatformBinaryHygiene({
  required File libraryFile,
  required OS targetOS,
  required Architecture targetArch,
  IOSSdk? targetIosSdk,
  String? expectedAndroidSoname,
  String? expectedAndroidNeeded,
}) async {
  if (targetOS == OS.android) {
    await _verifyAndroidElfHygiene(
      libraryFile: libraryFile,
      targetArch: targetArch,
      expectedSoname: expectedAndroidSoname,
      expectedNeeded: expectedAndroidNeeded,
    );
  } else if (targetOS == OS.iOS) {
    await _verifyIosMachOHygiene(
      libraryFile: libraryFile,
      targetArch: targetArch,
      targetIosSdk: targetIosSdk!,
    );
  }
}

Future<void> _verifyAndroidElfHygiene({
  required File libraryFile,
  required Architecture targetArch,
  String? expectedSoname,
  String? expectedNeeded,
}) async {
  final readelfExe = await _findFirstTool(['readelf', 'llvm-readelf']);
  if (readelfExe == null) {
    throw StateError(
      'Neither `readelf` nor `llvm-readelf` found to verify Android ELF '
      'binary ${libraryFile.path}.',
    );
  }

  // 1. Verify ELF machine type.
  final headerRes = await Process.run(readelfExe, ['-h', libraryFile.path]);
  if (headerRes.exitCode != 0) {
    throw StateError(
      '$readelfExe -h failed on ${libraryFile.path}: ${headerRes.stderr}',
    );
  }
  final headerOut = headerRes.stdout as String;
  final expectedMachine = switch (targetArch) {
    Architecture.arm64 => 'AArch64',
    Architecture.x64 => 'X86-64',
    Architecture.arm => 'ARM',
    Architecture.riscv64 => 'RISC-V',
    _ => null,
  };
  if (expectedMachine != null &&
      !headerOut.toLowerCase().contains(expectedMachine.toLowerCase())) {
    throw StateError(
      '${libraryFile.path}: expected ELF machine $expectedMachine, got:\n'
      '$headerOut',
    );
  }

  // 2. Verify all PT_LOAD segments have >= 16 KB (0x4000) alignment.
  final phRes = await Process.run(readelfExe, ['-lW', libraryFile.path]);
  if (phRes.exitCode != 0) {
    throw StateError(
      '$readelfExe -lW failed on ${libraryFile.path}: ${phRes.stderr}',
    );
  }
  final phOut = phRes.stdout as String;
  final alignments = parseAndroidLoadAlignments(phOut);
  if (alignments.isEmpty) {
    throw StateError(
      '${libraryFile.path}: no PT_LOAD segments found in readelf -lW output.',
    );
  }
  for (final align in alignments) {
    if (align < 0x4000) {
      throw StateError(
        '${libraryFile.path}: PT_LOAD segment alignment 0x${align.toRadixString(16)} '
        'is below the required 16 KB (0x4000) Android page-size alignment.',
      );
    }
  }

  // 3. Verify DT_NEEDED / DT_SONAME hygiene.
  final dynRes = await Process.run(readelfExe, ['-d', libraryFile.path]);
  if (dynRes.exitCode != 0) {
    throw StateError(
      '$readelfExe -d failed on ${libraryFile.path}: ${dynRes.stderr}',
    );
  }
  final dynOut = dynRes.stdout as String;
  final (:needed, :soname) = parseElfDynamicEntries(dynOut);
  const forbiddenNeeded = ['libc++_shared.so', 'libgfortran', 'libpthread.so'];
  for (final dep in needed) {
    for (final forbidden in forbiddenNeeded) {
      if (dep.contains(forbidden)) {
        throw StateError(
          '${libraryFile.path}: forbidden Android DT_NEEDED dependency "$dep" '
          '(all needed: $needed).',
        );
      }
    }
  }
  if (expectedSoname != null && soname != expectedSoname) {
    throw StateError(
      '${libraryFile.path}: expected DT_SONAME "$expectedSoname", got "$soname".',
    );
  }
  if (expectedNeeded != null && !needed.contains(expectedNeeded)) {
    throw StateError(
      '${libraryFile.path}: expected DT_NEEDED "$expectedNeeded", got $needed.',
    );
  }
}

/// Parses `PT_LOAD` alignment values from `readelf -lW` output.
List<int> parseAndroidLoadAlignments(String readelfProgramHeaders) {
  final alignments = <int>[];
  for (final rawLine in readelfProgramHeaders.split('\n')) {
    final line = rawLine.trim();
    if (!line.startsWith('LOAD')) continue;
    final tokens = line.split(RegExp(r'\s+'));
    if (tokens.length < 7) continue;
    final alignToken = tokens.last;
    final parsed = alignToken.startsWith('0x')
        ? int.tryParse(alignToken.substring(2), radix: 16)
        : int.tryParse(alignToken);
    if (parsed != null) {
      alignments.add(parsed);
    }
  }
  return alignments;
}

/// Parses `NEEDED` and `SONAME` entries from `readelf -d` output.
({List<String> needed, String? soname}) parseElfDynamicEntries(
  String readelfDynamicOutput,
) {
  final needed = <String>[];
  String? soname;
  final neededRe = RegExp(r'\(NEEDED\)\s+Shared library:\s+\[([^\]]+)\]');
  final sonameRe = RegExp(r'\(SONAME\)\s+Library soname:\s+\[([^\]]+)\]');
  for (final line in readelfDynamicOutput.split('\n')) {
    final nMatch = neededRe.firstMatch(line);
    if (nMatch != null) {
      needed.add(nMatch.group(1)!);
    }
    final sMatch = sonameRe.firstMatch(line);
    if (sMatch != null) {
      soname = sMatch.group(1)!;
    }
  }
  return (needed: needed, soname: soname);
}

Future<void> _verifyIosMachOHygiene({
  required File libraryFile,
  required Architecture targetArch,
  required IOSSdk targetIosSdk,
}) async {
  // 1. Verify slice architecture via `lipo -archs`.
  final lipoRes = await Process.run('lipo', ['-archs', libraryFile.path]);
  if (lipoRes.exitCode != 0) {
    throw StateError(
      'lipo -archs failed on ${libraryFile.path}: ${lipoRes.stderr}',
    );
  }
  final expectedArch = targetArch == Architecture.arm64 ? 'arm64' : 'x86_64';
  final archs = (lipoRes.stdout as String).trim().split(RegExp(r'\s+'));
  if (!archs.contains(expectedArch)) {
    throw StateError(
      '${libraryFile.path}: expected Mach-O arch "$expectedArch", got $archs.',
    );
  }

  // 2. Verify LC_BUILD_VERSION platform, LC_ID_DYLIB, and absence of LC_REEXPORT_DYLIB.
  final otoolLoadRes = await Process.run('otool', ['-l', libraryFile.path]);
  if (otoolLoadRes.exitCode != 0) {
    throw StateError(
      'otool -l failed on ${libraryFile.path}: ${otoolLoadRes.stderr}',
    );
  }
  final loadOut = otoolLoadRes.stdout as String;
  verifyIosMachOLoadCommands(
    loadCommandsOutput: loadOut,
    libraryPath: libraryFile.path,
    targetIosSdk: targetIosSdk,
  );

  // 3. Verify `otool -L` has no host Homebrew/usr-local paths.
  final otoolDepsRes = await Process.run('otool', ['-L', libraryFile.path]);
  if (otoolDepsRes.exitCode != 0) {
    throw StateError(
      'otool -L failed on ${libraryFile.path}: ${otoolDepsRes.stderr}',
    );
  }
  final depsOut = otoolDepsRes.stdout as String;
  for (final rawLine in depsOut.split('\n').skip(1)) {
    final line = rawLine.trim();
    if (line.isEmpty) continue;
    if (line.startsWith('/opt/homebrew/') ||
        line.startsWith('/usr/local/') ||
        line.contains('.dart_tool/')) {
      throw StateError(
        '${libraryFile.path}: unexpected host library reference in otool -L: $line',
      );
    }
  }
}

/// Validates `otool -l` output for an iOS dynamic library.
void verifyIosMachOLoadCommands({
  required String loadCommandsOutput,
  required String libraryPath,
  required IOSSdk targetIosSdk,
}) {
  final expectedPlatform = targetIosSdk == IOSSdk.iPhoneOS
      ? 'IOS'
      : 'IOSSIMULATOR';
  final expectedPlatformId = targetIosSdk == IOSSdk.iPhoneOS ? '2' : '7';

  final buildVersionBlock = RegExp(
    r'cmd LC_BUILD_VERSION[\s\S]*?(?=\nLoad command|$)',
  ).firstMatch(loadCommandsOutput);
  if (buildVersionBlock == null) {
    throw StateError(
      '$libraryPath: missing LC_BUILD_VERSION load command in otool -l output.',
    );
  }
  final platformMatch = RegExp(
    r'platform\s+(\S+)',
  ).firstMatch(buildVersionBlock.group(0)!);
  final platform = platformMatch?.group(1)?.toUpperCase();
  if (platform != expectedPlatform && platform != expectedPlatformId) {
    throw StateError(
      '$libraryPath: expected LC_BUILD_VERSION platform $expectedPlatform '
      '(or $expectedPlatformId) for $targetIosSdk, got "$platform".',
    );
  }

  final idBlock = RegExp(
    r'cmd LC_ID_DYLIB[\s\S]*?(?=\nLoad command|$)',
  ).firstMatch(loadCommandsOutput);
  final idName = idBlock == null
      ? null
      : RegExp(r'name\s+(\S+)').firstMatch(idBlock.group(0)!)?.group(1);
  if (idName == null || !idName.startsWith('@rpath/')) {
    throw StateError(
      '$libraryPath: expected @rpath/ LC_ID_DYLIB install_name, got "$idName".',
    );
  }

  if (loadCommandsOutput.contains('LC_REEXPORT_DYLIB') &&
      loadCommandsOutput.contains('Accelerate.framework')) {
    throw StateError(
      '$libraryPath: unexpected LC_REEXPORT_DYLIB of Accelerate.framework; '
      'link -framework Accelerate directly instead.',
    );
  }
}

Future<String?> _findFirstTool(List<String> candidates) async {
  for (final tool in candidates) {
    try {
      final res = await Process.run(tool, ['--version']);
      if (res.exitCode == 0) return tool;
    } on ProcessException {
      continue;
    }
  }
  return null;
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
      ('readelf', ['-Ws', '--dyn-syms', libraryFile.path]),
      ('llvm-nm', ['-D', '--defined-only', libraryFile.path]),
      ('nm', ['-D', '--defined-only', libraryFile.path]),
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
      r'^\d+:\s+[0-9a-fA-F]+\s+(?:0x[0-9a-fA-F]+|\d+)\s+\S+\s+(?:GLOBAL|WEAK)\s+\S+\s+(?!UND)\S+\s+(\S+)(?:\s+\(\d+\))?$',
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
