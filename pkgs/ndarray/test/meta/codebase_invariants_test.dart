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

import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/features.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/dart/element/type.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/hook_helpers/hashes.dart';
import 'package:test/test.dart';

Directory _findPackageRoot() {
  var dir = Directory.current;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/hook').existsSync()) {
      return dir;
    }
    final sub = Directory('${dir.path}/pkgs/ndarray');
    if (sub.existsSync()) {
      return sub;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError('Could not locate pkgs/ndarray root directory.');
    }
    dir = parent;
  }
}

/// Converts [path] to a normalized native path, as required by the analyzer
/// (on Windows, paths built with `'${dir.path}/x'` mix separators).
String _native(String path) => Uri.file(path).toFilePath();

/// Converts [path] to use `/` separators, for platform-independent substring
/// checks such as `contains('/src/operations/')`.
String _posix(String path) => path.replaceAll(r'\', '/');

List<File> _dartFilesIn(Directory dir) {
  if (!dir.existsSync()) return const [];
  return dir
      .listSync(recursive: true)
      .whereType<File>()
      .where((f) => f.path.endsWith('.dart'))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));
}

String _stripCppComments(String source) {
  final withoutBlock = source.replaceAll(
    RegExp(r'/\*[\s\S]*?\*/', multiLine: true),
    '',
  );
  return withoutBlock
      .split('\n')
      .map((line) {
        final idx = line.indexOf('//');
        return idx >= 0 ? line.substring(0, idx) : line;
      })
      .join('\n');
}

void main() {
  final pkgRoot = _findPackageRoot();
  final monorepoRoot = pkgRoot.parent.parent;
  final pkgsDir = Directory('${monorepoRoot.path}/pkgs');
  final libDir = Directory('${pkgRoot.path}/lib');
  final hookDir = Directory('${pkgRoot.path}/hook');
  final libFiles = _dartFilesIn(libDir);
  final featureSet = FeatureSet.latestLanguageVersion();

  group('Codebase & FFI Invariants', () {
    test(
      'Every ScratchArena.marker in lib/ is paired with try / finally ScratchArena.reset',
      () {
        final violations = <String>[];

        for (final file in libFiles) {
          if (file.path.endsWith('scratch_arena.dart')) continue;
          final result = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _ScratchArenaVisitor(file.path, result.lineInfo);
          result.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Every `final marker = ScratchArena.marker;` must be followed by '
              'a `try { ... } finally { ScratchArena.reset(marker); }` block:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'Every int-returning @ffi.Native call in lib/ (including extensions) checks its return code',
      () {
        final bindingFiles = [
          File('${libDir.path}/src/ndarray_bindings.dart'),
          File('${libDir.path}/src/ndarray_extensions_bindings.dart'),
        ];

        final intReturningNativeFunctions = <String>{};
        for (final bindingsFile in bindingFiles) {
          if (!bindingsFile.existsSync()) continue;
          final bindingsUnit = parseFile(
            path: _native(bindingsFile.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          ).unit;

          for (final decl in bindingsUnit.declarations) {
            if (decl is FunctionDeclaration) {
              final name = decl.name.lexeme;
              final returnType = decl.returnType?.toSource();
              if (name.startsWith('native_') && returnType == 'int') {
                intReturningNativeFunctions.add(name);
              }
            }
          }
        }

        expect(
          intReturningNativeFunctions,
          isNotEmpty,
          reason: 'Expected to discover int-returning native_* FFI bindings.',
        );

        final uncheckedCalls = <String>[];
        for (final file in libFiles) {
          if (file.path.endsWith('ndarray_bindings.dart') ||
              file.path.endsWith('ndarray_extensions_bindings.dart')) {
            continue;
          }
          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _UncheckedNativeCallVisitor(
            file.path,
            parsed.lineInfo,
            intReturningNativeFunctions,
          );
          parsed.unit.accept(visitor);
          uncheckedCalls.addAll(visitor.violations);
        }

        expect(
          uncheckedCalls,
          isEmpty,
          reason:
              'All int-returning native_* FFI functions can fail (e.g. OOM -4 '
              'or bounds errors) and must have their return code checked:\n'
              '${uncheckedCalls.join('\n')}',
        );
      },
    );

    test(
      'NativeFinalizer.attach never passes externalSize across all workspace packages',
      () {
        final violations = <String>[];
        final allPkgLibFiles = pkgsDir.existsSync()
            ? pkgsDir
                  .listSync()
                  .whereType<Directory>()
                  .expand((d) => _dartFilesIn(Directory('${d.path}/lib')))
                  .toList()
            : libFiles;

        for (final file in allPkgLibFiles) {
          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _FinalizerExternalSizeVisitor(
            file.path,
            parsed.lineInfo,
          );
          parsed.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Per Dart VM team guidance, do not pass externalSize to '
              'NativeFinalizer.attach; rely on NDArray.scope / dispose() instead:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'C++ hook sources are exception-free (-fno-exceptions safe), include guards exist, and exported functions match headers',
      () {
        final cppFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.cpp'))
            .toList();
        final headerFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.h'))
            .toList();
        expect(cppFiles, isNotEmpty);
        expect(headerFiles, isNotEmpty);

        final violations = <String>[];

        for (final hFile in headerFiles) {
          final raw = hFile.readAsStringSync();
          final baseName = hFile.uri.pathSegments.last;
          if (!raw.contains('#pragma once') && !raw.contains('#ifndef')) {
            violations.add(
              '$baseName: missing `#pragma once` or `#ifndef` include guard.',
            );
          }
        }

        for (final cppFile in cppFiles) {
          final raw = cppFile.readAsStringSync();
          final stripped = _stripCppComments(raw);
          final baseName = cppFile.uri.pathSegments.last;

          if (RegExp(r'\bstd::vector\b').hasMatch(stripped)) {
            violations.add(
              '$baseName: contains `std::vector` (use `NoThrowBuffer` with `std::nothrow` under `-fno-exceptions`).',
            );
          }
          if (RegExp(r'\bstd::call_once\b').hasMatch(stripped)) {
            violations.add(
              '$baseName: contains `std::call_once` (can throw `std::system_error`; use `std::atomic` instead).',
            );
          }
          if (RegExp(r'\bthrow\b').hasMatch(stripped)) {
            violations.add(
              '$baseName: contains `throw` statement under `-fno-exceptions`.',
            );
          }
          if (stripped.contains('std::memcpy') &&
              !raw.contains('<cstring>') &&
              !raw.contains('<string.h>')) {
            violations.add(
              '$baseName: uses `std::memcpy` without `#include <cstring>`.',
            );
          }

          // Check that every NDARRAY_EXPORT / FFI_PLUGIN_EXPORT function in foo.cpp is declared in foo.h
          final headerPath = cppFile.path.replaceFirst(RegExp(r'\.cpp$'), '.h');
          final headerFile = File(headerPath);
          if (headerFile.existsSync()) {
            final headerSource = _stripCppComments(
              headerFile.readAsStringSync(),
            );
            final exportRegex = RegExp(
              r'(?:NDARRAY_EXPORT|FFI_PLUGIN_EXPORT)\s+[A-Za-z0-9_*\s]+\s+([A-Za-z0-9_]+)\s*\(',
            );
            for (final match in exportRegex.allMatches(stripped)) {
              final fnName = match.group(1)!;
              if (!RegExp('\\b$fnName\\b').hasMatch(headerSource)) {
                violations.add(
                  '$baseName: exported function `$fnName` is missing from ${headerFile.uri.pathSegments.last}.',
                );
              }
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'C++ hook files must be strictly `-fno-exceptions` clean and '
              'declare all exported symbols in their header:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'All public top-level API declarations transitively exported by lib/ndarray.dart have Dartdoc comments',
      () {
        final entrypoint = File('${libDir.path}/ndarray.dart');
        final visited = <String>{};
        final exportedFiles = <(File, Set<String>?, Set<String>?)>[];

        void collectExports(
          File file, {
          Set<String>? showNames,
          Set<String>? hideNames,
        }) {
          final canonical = file.resolveSymbolicLinksSync();
          if (!visited.add('$canonical|$showNames|$hideNames')) return;
          exportedFiles.add((file, showNames, hideNames));

          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          for (final directive in parsed.unit.directives) {
            if (directive is ExportDirective) {
              final uriStr = directive.uri.stringValue;
              if (uriStr == null || uriStr.startsWith('package:')) continue;
              final resolved = File('${file.parent.path}/$uriStr');
              if (!resolved.existsSync()) continue;

              Set<String>? childShow = showNames;
              final childHide = <String>{...?hideNames};
              for (final combinator in directive.combinators) {
                if (combinator is ShowCombinator) {
                  final names = combinator.shownNames
                      .map((n) => n.name)
                      .toSet();
                  childShow = childShow == null
                      ? names
                      : childShow.intersection(names);
                } else if (combinator is HideCombinator) {
                  childHide.addAll(combinator.hiddenNames.map((n) => n.name));
                }
              }
              collectExports(
                resolved,
                showNames: childShow,
                hideNames: childHide,
              );
            }
          }
        }

        collectExports(entrypoint);
        expect(exportedFiles.length, greaterThan(20));

        final missingDocs = <String>[];

        for (final (file, showNames, hideNames) in exportedFiles) {
          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          for (final decl in parsed.unit.declarations) {
            final name = switch (decl) {
              FunctionDeclaration(:final name) => name.lexeme,
              ClassDeclaration(:final namePart) => namePart.typeName.lexeme,
              EnumDeclaration(:final namePart) => namePart.typeName.lexeme,
              MixinDeclaration(:final name) => name.lexeme,
              GenericTypeAlias(:final name) => name.lexeme,
              TopLevelVariableDeclaration(:final variables) =>
                variables.variables.first.name.lexeme,
              _ => null,
            };
            if (name == null || name.startsWith('_')) continue;
            if (showNames != null && !showNames.contains(name)) continue;
            if (hideNames != null && hideNames.contains(name)) continue;
            if (name.endsWith('_helper')) continue;

            final isInternalAnnotated = decl.metadata.any(
              (m) => m.name.name == 'internal',
            );
            if (isInternalAnnotated) continue;

            if (decl.documentationComment == null) {
              final line = parsed.lineInfo.getLocation(decl.offset).lineNumber;
              missingDocs.add(
                '${file.path}:$line — `$name` is missing /// dartdoc',
              );
            }
          }
        }

        expect(
          missingDocs,
          isEmpty,
          reason:
              'Every public top-level symbol transitively exported by lib/ndarray.dart must have a `///` dartdoc comment:\n'
              '${missingDocs.join('\n')}',
        );
      },
    );

    test(
      'Architecture & memory invariants: no cross-package src/ imports, no external .dataRaw, no pointer+offsetElements double-offset, no CellFlat+getIndex',
      () {
        final violations = <String>[];

        // 1. No cross-package package:<other>/src/ imports across workspace packages in pkgs/*/lib/
        if (pkgsDir.existsSync()) {
          final workspacePackages = pkgsDir
              .listSync()
              .whereType<Directory>()
              .map((d) => d.uri.pathSegments.where((s) => s.isNotEmpty).last)
              .toSet();
          for (final pkg in pkgsDir.listSync().whereType<Directory>()) {
            final pkgName = pkg.uri.pathSegments
                .where((s) => s.isNotEmpty)
                .last;
            for (final dartFile in _dartFilesIn(Directory('${pkg.path}/lib'))) {
              final lines = dartFile.readAsLinesSync();
              for (var i = 0; i < lines.length; i++) {
                final m = RegExp(
                  r'''^\s*(?:import|export)\s+['"]package:([^/]+)/src/''',
                ).firstMatch(lines[i]);
                if (m != null) {
                  final importedPkg = m.group(1)!;
                  if (importedPkg != pkgName &&
                      workspacePackages.contains(importedPkg)) {
                    violations.add(
                      '${dartFile.path}:${i + 1} — cross-package src/ import: `${lines[i].trim()}`',
                    );
                  }
                }
              }
            }
          }
        }

        // 2. Check ndarray/lib/ files
        for (final file in libFiles) {
          final isNdarrayCore = _posix(file.path).endsWith('/src/ndarray.dart');
          final isOperations = _posix(file.path).contains('/src/operations/');
          final lines = file.readAsLinesSync();

          for (var i = 0; i < lines.length; i++) {
            final line = lines[i];
            if (!isNdarrayCore && RegExp(r'\bdataRaw\b').hasMatch(line)) {
              violations.add(
                '${file.path}:${i + 1} — `.dataRaw` accessed outside ndarray.dart',
              );
            }
            if (isOperations &&
                line.contains('.pointer') &&
                line.contains('.offsetElements')) {
              violations.add(
                '${file.path}:${i + 1} — `.pointer` and `.offsetElements` on the same line (risk of double-offsetting)',
              );
            }
            if ((line.contains('getCellFlat') ||
                    line.contains('setCellFlat')) &&
                line.contains('getIndex')) {
              violations.add(
                '${file.path}:${i + 1} — `getCellFlat`/`setCellFlat` used with `iter.getIndex` (must use `getCellRaw`/`setCellRaw`)',
              );
            }
            if (isOperations && line.contains('.setRange(')) {
              violations.add(
                '${file.path}:${i + 1} — `.setRange(` used in operations/ (use NDArray views and `.copy()` instead)',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Architectural and memory access invariants violated:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'DType enum indices and native package release versions / recursive source hooks stay in sync',
      () {
        final indexingH = File(
          '${hookDir.path}/custom_indexing.h',
        ).readAsStringSync();
        final sortingH = File(
          '${hookDir.path}/custom_sorting.h',
        ).readAsStringSync();
        final expectedMacros = <DType, String>{
          DType.float64: 'DTYPE_FLOAT64',
          DType.float32: 'DTYPE_FLOAT32',
          DType.float16: 'DTYPE_FLOAT16',
          DType.bfloat16: 'DTYPE_BFLOAT16',
          DType.int64: 'DTYPE_INT64',
          DType.int32: 'DTYPE_INT32',
          DType.int16: 'DTYPE_INT16',
          DType.int8: 'DTYPE_INT8',
          DType.uint64: 'DTYPE_UINT64',
          DType.uint32: 'DTYPE_UINT32',
          DType.uint16: 'DTYPE_UINT16',
          DType.uint8: 'DTYPE_UINT8',
          DType.complex128: 'DTYPE_COMPLEX128',
          DType.complex64: 'DTYPE_COMPLEX64',
          DType.boolean: 'DTYPE_BOOLEAN',
        };

        expect(expectedMacros.length, equals(DType.values.length));
        for (final entry in expectedMacros.entries) {
          final idx = entry.key.index;
          final macro = entry.value;
          final pattern = RegExp('#define\\s+$macro\\s+$idx\\b');
          expect(
            pattern.hasMatch(indexingH) && pattern.hasMatch(sortingH),
            isTrue,
            reason:
                'Expected `#define $macro $idx` in custom_indexing.h and custom_sorting.h',
          );
        }

        // Check native packages (ndarray, openblas, pocketfft, symengine)
        if (pkgsDir.existsSync()) {
          for (final pkgName in [
            'ndarray',
            'openblas',
            'pocketfft',
            'symengine',
          ]) {
            final dir = Directory('${pkgsDir.path}/$pkgName');
            final hashesFile = File(
              '${dir.path}/lib/src/hook_helpers/hashes.dart',
            );
            if (!hashesFile.existsSync()) continue;

            final pubspec = File('${dir.path}/pubspec.yaml').readAsStringSync();
            final pubVersion = RegExp(
              r'^version:\s*(\S+)',
              multiLine: true,
            ).firstMatch(pubspec)?.group(1);
            final hashesTxt = hashesFile.readAsStringSync();
            final tagVersion = RegExp(
              r"const\s+version\s*=\s*'artifacts-v([^']+)';",
            ).firstMatch(hashesTxt)?.group(1);
            expect(
              tagVersion,
              isNotNull,
              reason:
                  '$pkgName: hashes.dart must define `const version = \'artifacts-v<version>\';`.',
            );
            final pinnedSourceHash = RegExp(
              r"const\s+nativeSourceHash\s*=\s*'([0-9a-f]{64})';",
            ).firstMatch(hashesTxt)?.group(1);
            final currentSourceHash = computeNativeSourceHash(dir.uri);
            if (pinnedSourceHash == currentSourceHash && pkgName == 'ndarray') {
              expect(
                tagVersion,
                equals(pubVersion),
                reason:
                    '$pkgName: hashes.dart version (`artifacts-v$tagVersion`) must match pubspec.yaml version (`$pubVersion`).',
              );
            }

            // Hash inputs in hook/ must only be top-level tracked files (never recursive).
            expect(
              hashesTxt.contains('listSync(recursive: true)'),
              isFalse,
              reason:
                  '$pkgName: nativeSourceFiles in hashes.dart must not use `listSync(recursive: true)` '
                  'so downloaded or untracked subdirectories under hook/ cannot pollute nativeSourceHash.',
            );
            final subDirs = Directory(
              '${dir.path}/hook',
            ).listSync().whereType<Directory>().toList();
            expect(
              subDirs,
              isEmpty,
              reason:
                  '$pkgName: hook/ must not contain subdirectories (${subDirs.map((d) => d.path).toList()}); '
                  'store downloaded third-party sources in outputDirectoryShared.',
            );
          }
        }
      },
    );

    test(
      'Zero duplicate @ffi.Native external declarations across binding files (S1)',
      () {
        final bindingFiles = [
          File('${libDir.path}/src/ndarray_bindings.dart'),
          File('${libDir.path}/src/ndarray_extensions_bindings.dart'),
        ];
        final seen = <String, String>{};
        final duplicates = <String>[];

        for (final file in bindingFiles) {
          if (!file.existsSync()) continue;
          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final baseName = file.uri.pathSegments.last;
          for (final decl in parsed.unit.declarations) {
            if (decl is FunctionDeclaration && decl.externalKeyword != null) {
              final name = decl.name.lexeme;
              final line = parsed.lineInfo.getLocation(decl.offset).lineNumber;
              final loc = '$baseName:$line';
              if (seen.containsKey(name)) {
                duplicates.add(
                  '`$name` declared at both ${seen[name]} and $loc',
                );
              } else {
                seen[name] = loc;
              }
            }
          }
        }

        expect(
          duplicates,
          isEmpty,
          reason:
              'Duplicate external @ffi.Native declarations risk signature drift:\n'
              '${duplicates.join('\n')}',
        );
      },
    );

    test(
      'ScratchArena.getStridedBuffer segment bounds and integer floordiv error checks (S6)',
      () {
        final violations = <String>[];

        for (final file in libFiles) {
          final content = file.readAsStringSync();
          final lines = content.split('\n');

          // 1. Check getStridedBuffer(ndim, [segments]) vs cBuffer + ndim * k
          final stridedDeclRegex = RegExp(
            r'final\s+(\w+)\s*=\s*ScratchArena\.getStridedBuffer\(\s*(\w+)(?:\s*,\s*(\d+))?\s*\)',
          );
          for (var i = 0; i < lines.length; i++) {
            final m = stridedDeclRegex.firstMatch(lines[i]);
            if (m != null) {
              final bufVar = m.group(1)!;
              final ndimVar = m.group(2)!;
              final segments = int.parse(m.group(3) ?? '3');
              final endLine = (i + 60 < lines.length) ? i + 60 : lines.length;
              final window = lines.sublist(i, endLine).join('\n');
              final offsetRegex = RegExp(
                '$bufVar\\s*\\+\\s*$ndimVar\\s*\\*\\s*(\\d+)',
              );
              for (final om in offsetRegex.allMatches(window)) {
                final k = int.parse(om.group(1)!);
                if (k >= segments) {
                  violations.add(
                    '${file.path}:${i + 1} — `$bufVar + $ndimVar * $k` exceeds allocated segments ($segments) in `getStridedBuffer`.',
                  );
                }
              }
            }
          }

          // 2. Every file invoking v_floordiv_int* or s_floordiv_int* must check get_and_reset_division_error()
          if (!file.path.endsWith('ndarray_bindings.dart') &&
              RegExp(r'\b[vs]_floordiv_int').hasMatch(content)) {
            if (!content.contains('get_and_reset_division_error()')) {
              violations.add(
                '${file.path} — calls `v_floordiv_int*` / `s_floordiv_int*` without checking `get_and_reset_division_error()`.',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Strided scratch buffer segment or integer division error check violation:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'C++ hook sources use std::nothrow on all heap allocations and avoid std::stable_sort/std::map/iostream/printf (S7)',
      () {
        final cppAndHeaderFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.cpp') || f.path.endsWith('.h'))
            .toList();
        final violations = <String>[];

        for (final file in cppAndHeaderFiles) {
          final raw = file.readAsStringSync();
          final stripped = _stripCppComments(raw);
          final baseName = file.uri.pathSegments.last;

          // Check raw `new` without `(std::nothrow)`
          for (final m in RegExp(r'\bnew\b([^\n;]*)').allMatches(stripped)) {
            final tail = m.group(1)!;
            if (!tail.contains('nothrow')) {
              violations.add(
                '$baseName — raw `new` without `(std::nothrow)`: `new$tail`',
              );
            }
          }

          if (RegExp(
            r'\bstd::(?:unordered_)?(?:map|set)\b',
          ).hasMatch(stripped)) {
            violations.add(
              '$baseName — contains `std::map`/`std::set` (throws `std::bad_alloc` under `-fno-exceptions`).',
            );
          }
          if (raw.contains('#include <iostream>')) {
            violations.add(
              '$baseName — includes `<iostream>` (adds static constructors and I/O bloat).',
            );
          }
          if (RegExp(
            r'\b(?:printf|std::cout|std::cerr)\b',
          ).hasMatch(stripped)) {
            violations.add(
              '$baseName — contains `printf`/`std::cout`/`std::cerr` in FFI library.',
            );
          }
          for (final m in RegExp(
            r'\bstd::(?:fmodf|floorf|ceilf|fabsf|sqrtf|sinf|cosf|tanf|asinf|acosf|atanf|atan2f|sinhf|coshf|tanhf|expf|logf|log10f|powf|ldexpf|frexpf|modff|rintf|hypotf|copysignf)\b',
          ).allMatches(stripped)) {
            violations.add(
              '$baseName — contains non-portable `${m.group(0)}` (not in `namespace std` on all libstdc++ versions); use the overloaded `std::` name without the `f` suffix.',
            );
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'C++ `-fno-exceptions` / heap / std:: math portability discipline violations:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'C++ hook sources use 64-bit file offsets and avoid 32-bit long declarations (LLP64 safe)',
      () {
        final cppAndHeaderFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.cpp') || f.path.endsWith('.h'))
            .toList();
        final violations = <String>[];

        for (final file in cppAndHeaderFiles) {
          final raw = file.readAsStringSync();
          final stripped = _stripCppComments(raw);
          final codeOnly = stripped.replaceAll(
            RegExp(r'"(?:\\.|[^"\\])*"'),
            '""',
          );
          final baseName = file.uri.pathSegments.last;

          // Check for ftell( and fseek(
          for (final m in RegExp(
            r'\b(fseek|ftell)\s*\(',
          ).allMatches(codeOnly)) {
            violations.add(
              '$baseName — contains 32-bit `${m.group(1)}(`, require 64-bit npz_fseek64/npz_ftell64 or _fseeki64/ftello.',
            );
          }

          // Check for bare `long` (allow `long long` and `unsigned long long`)
          final bareLongRegex = RegExp(r'(?<!\blong\s+)\blong\b(?!\s+long\b)');
          final lines = codeOnly.split('\n');
          for (var i = 0; i < lines.length; i++) {
            final line = lines[i];
            if (bareLongRegex.hasMatch(line)) {
              violations.add(
                '$baseName:${i + 1} — contains bare `long` declaration/cast (32-bit on Windows LLP64): `${line.trim()}`',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'C++ 64-bit file offset / LLP64 integer width violations:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'nativeSourceHash matches the current hook/ sources only if they are unchanged since the release tag',
      () {
        // The pin records the hook/ sources that the prebuilt [version]
        // binaries were built from. The build hook falls back to building from
        // source when the current sources hash differently, so editing the pin
        // to match unreleased sources makes consumers download stale binaries.
        // The pin must only be written by
        // `dart tool/regenerate_hashes.dart <release-tag>`.
        final tagCheck = Process.runSync('git', [
          'rev-parse',
          '--verify',
          '--quiet',
          'refs/tags/$version',
        ], workingDirectory: pkgRoot.path);
        if (tagCheck.exitCode != 0) {
          markTestSkipped(
            'Release tag $version is not available locally; run `git fetch --tags`.',
          );
          return;
        }
        final diff = Process.runSync('git', [
          'diff',
          '--quiet',
          version,
          '--',
          for (final extension in ['dart', 'c', 'cpp', 'h', 'def'])
            ':(glob)hook/*.$extension',
        ], workingDirectory: pkgRoot.path);
        expect(
          diff.exitCode,
          anyOf(0, 1),
          reason: 'git diff failed: ${diff.stderr}',
        );
        final sourcesChangedSinceRelease = diff.exitCode == 1;
        final pinMatchesCurrentSources =
            nativeSourceHash == computeNativeSourceHash(pkgRoot.uri);
        expect(
          pinMatchesCurrentSources,
          !sourcesChangedSinceRelease,
          reason: sourcesChangedSinceRelease
              ? 'hook/ sources changed since release $version, but '
                    'nativeSourceHash claims they match the prebuilt binaries. '
                    'That disables the build-from-source fallback and ships '
                    'stale binaries. Keep the pin, or cut a new release and run '
                    '`dart tool/regenerate_hashes.dart <tag>`.'
              : 'hook/ sources are unchanged since release $version, but '
                    'nativeSourceHash does not match them. Regenerate it with '
                    '`dart tool/regenerate_hashes.dart $version`.',
        );
      },
    );

    test(
      'Public functions with an `out` parameter use a named parameter `{NDArray? out}` and BinaryOp/UnaryOp enums are exhaustively dispatched (S2 & S9)',
      () {
        final violations = <String>[];

        // 1. All public functions in lib/src/operations/ with a parameter named `out` must declare it as a named parameter
        for (final file in libFiles) {
          if (!_posix(file.path).contains('/src/operations/')) continue;
          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          for (final decl in parsed.unit.declarations) {
            if (decl is FunctionDeclaration) {
              final fnName = decl.name.lexeme;
              if (fnName.startsWith('_') || fnName == 'where') continue;
              final params = decl.functionExpression.parameters?.parameters;
              if (params == null) continue;
              for (final p in params) {
                if (p.name?.lexeme == 'out' && !p.isNamed) {
                  final line = parsed.lineInfo.getLocation(p.offset).lineNumber;
                  violations.add(
                    '${file.path}:$line — `$fnName` declares `out` as a positional parameter instead of a named parameter.',
                  );
                }
              }
            }
          }
        }

        // 2. Every reducible BinaryOp and every UnaryOp in binary_op.dart is dispatched in ufunc_methods.dart
        final binOpFile = File(
          '${libDir.path}/src/operations/math/binary_op.dart',
        );
        final ufuncMethodsTxt = File(
          '${libDir.path}/src/operations/math/ufunc_methods.dart',
        ).readAsStringSync();
        for (final op in BinaryOp.values.where((o) => o.isReducible)) {
          if (!ufuncMethodsTxt.contains('BinaryOp.${op.name}')) {
            violations.add(
              'BinaryOp.${op.name} (isReducible: true) is not dispatched in ufunc_methods.dart',
            );
          }
        }
        final binParsed = parseFile(
          path: _native(binOpFile.path),
          featureSet: featureSet,
          throwIfDiagnostics: false,
        );
        for (final decl in binParsed.unit.declarations) {
          if (decl is EnumDeclaration &&
              decl.namePart.typeName.lexeme == 'UnaryOp') {
            for (final c in decl.body.constants) {
              final cName = c.name.lexeme;
              if (!ufuncMethodsTxt.contains('UnaryOp.$cName')) {
                violations.add(
                  'UnaryOp.$cName is declared in binary_op.dart but not dispatched in ufunc_methods.dart',
                );
              }
            }
          }
        }

        // 3. Zero-violation ratchets: no NDArray<double|int|num|bool|Object|dynamic>, no print( in lib/, no solo_test in test/
        final badTypeGeneric = RegExp(
          r'\bNDArray<\s*(?:double|int|num|bool|Object|dynamic)\s*>',
        );
        for (final file in libFiles) {
          final lines = file.readAsLinesSync();
          for (var i = 0; i < lines.length; i++) {
            final line = lines[i];
            if (line.trimLeft().startsWith('//')) continue;
            if (badTypeGeneric.hasMatch(line)) {
              violations.add(
                '${file.path}:${i + 1} — uses Dart primitive type parameter on NDArray instead of DataType marker (`Float64`, `Int64`, etc.): `${line.trim()}`',
              );
            }
            if (RegExp(r'\bprint\s*\(').hasMatch(line)) {
              violations.add(
                '${file.path}:${i + 1} — `print(...)` statement in library code.',
              );
            }
          }
        }

        final testFiles = _dartFilesIn(Directory('${pkgRoot.path}/test'));
        for (final file in testFiles) {
          if (file.path.endsWith('codebase_invariants_test.dart')) continue;
          final content = file.readAsStringSync();
          if (RegExp(r'\bsolo_test\s*\(').hasMatch(content) ||
              RegExp(r'\bsolo_group\s*\(').hasMatch(content)) {
            violations.add(
              '${file.path} — contains `solo_test` or `solo_group`.',
            );
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'API signature, BinaryOp enum parity, or code hygiene ratchet violated:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'Resolved Semantic AST (DartType & Element) Invariants: exportNamespace type-closure, class modifiers/member dartdocs, receiver-type-aware NDArray.toList()/NDArray.data bans, ScratchArena.allocate<T> vs sizeOf<U> DartType equality, and zero dead extension bindings',
      () async {
        final resolvedLibPath = _native(libDir.resolveSymbolicLinksSync());
        final collection = AnalysisContextCollection(
          includedPaths: [resolvedLibPath],
        );
        final session = collection.contextFor(resolvedLibPath).currentSession;

        final libResult = await session.getResolvedLibrary(
          _native('$resolvedLibPath/ndarray.dart'),
        );
        expect(libResult, isA<ResolvedLibraryResult>());
        final exportNames = (libResult as ResolvedLibraryResult)
            .element
            .exportNamespace
            .definedNames2;
        expect(exportNames.length, greaterThan(350));

        final violations = <String>[];

        void checkNoRawNdarray(DartType type, String context) {
          if (type is InterfaceType) {
            if (type.element.name == 'NDArray' &&
                type.typeArguments.first is DynamicType) {
              violations.add(
                '$context — uses raw `NDArray<dynamic>` (`${type.getDisplayString()}`); specify `<T extends DTypeTag>` or a concrete `DTypeTag`.',
              );
            }
            for (final arg in type.typeArguments) {
              checkNoRawNdarray(arg, context);
            }
          } else if (type is RecordType) {
            for (final f in type.positionalFields) {
              checkNoRawNdarray(f.type, context);
            }
            for (final f in type.namedFields) {
              checkNoRawNdarray(f.type, context);
            }
          } else if (type is FunctionType) {
            checkNoRawNdarray(type.returnType, context);
            for (final p in type.formalParameters) {
              checkNoRawNdarray(p.type, context);
            }
          }
        }

        void checkNoPositionalRecordReturn(DartType type, String context) {
          if (type is RecordType) {
            if (type.positionalFields.isNotEmpty) {
              violations.add(
                '$context — returns a record with positional fields (`${type.getDisplayString()}`); use named record fields instead.',
              );
            }
            for (final f in type.namedFields) {
              checkNoPositionalRecordReturn(f.type, context);
            }
          } else if (type is InterfaceType) {
            for (final arg in type.typeArguments) {
              checkNoPositionalRecordReturn(arg, context);
            }
          }
        }

        // 1. Inspect resolved exportNamespace of package:ndarray/ndarray.dart
        for (final entry in exportNames.entries) {
          final name = entry.key;
          final el = entry.value;

          if (el.metadata.annotations.any((a) => a.isInternal)) {
            violations.add(
              'Exported symbol `$name` (${el.kind.displayName}) is annotated `@internal`.',
            );
          }

          if (el is ExecutableElement) {
            if (el.returnType is DynamicType && name != 'where') {
              violations.add(
                'Exported function `$name` returns untyped `dynamic`.',
              );
            }
            checkNoRawNdarray(el.returnType, 'Exported `$name` return type');
            checkNoPositionalRecordReturn(
              el.returnType,
              'Exported `$name` return type',
            );
            for (final p in el.formalParameters) {
              if (p.type is DynamicType && name != 'where') {
                violations.add(
                  'Exported `$name` parameter `${p.name}` uses untyped `dynamic`; use a generic type parameter or `Object`/`Object?`.',
                );
              }
              checkNoRawNdarray(
                p.type,
                'Exported `$name` parameter `${p.name}`',
              );
            }
          } else if (el is ClassElement) {
            if (!el.isFinal && !el.isSealed && !el.isAbstract && !el.isBase) {
              violations.add(
                'Exported class `${el.name}` must be marked `final`, `sealed`, `abstract`, or `base`.',
              );
            }
            for (final m in [...el.methods, ...el.getters]) {
              if (m.isPrivate || m.firstFragment.nameOffset == null) continue;
              if (m.metadata.annotations.any(
                (a) => a.isInternal || a.isOverride,
              )) {
                continue;
              }
              if (m.documentationComment == null) {
                violations.add(
                  'Public member `${el.name}.${m.name}` is missing `///` dartdoc.',
                );
              }
              checkNoRawNdarray(
                m.returnType,
                'Member `${el.name}.${m.name}` return type',
              );
              if (el.name == 'NDArray') {
                checkNoPositionalRecordReturn(
                  m.returnType,
                  'Member `${el.name}.${m.name}` return type',
                );
              }
              for (final p in m.formalParameters) {
                if (p.type is DynamicType) {
                  violations.add(
                    'Member `${el.name}.${m.name}` parameter `${p.name}` uses untyped `dynamic`; use a generic type parameter or `Object`/`Object?`.',
                  );
                }
                checkNoRawNdarray(
                  p.type,
                  'Member `${el.name}.${m.name}` parameter `${p.name}`',
                );
              }
            }
          }
        }

        // 2. Collect all @ffi.Native external functions in ndarray_extensions_bindings.dart
        final extBindingsRes = await session.getResolvedUnit(
          _native('$resolvedLibPath/src/ndarray_extensions_bindings.dart'),
        );
        final extFunctions = <ExecutableElement>{};
        if (extBindingsRes is ResolvedUnitResult) {
          for (final d in extBindingsRes.unit.declarations) {
            if (d is FunctionDeclaration && d.externalKeyword != null) {
              final el = d.declaredFragment?.element;
              if (el != null) extFunctions.add(el);
            }
          }
        }
        final usedExtFunctions = <ExecutableElement>{};

        // 3. Walk resolved AST of every implementation file in lib/
        for (final file in libFiles) {
          if (file.path.endsWith('ndarray_bindings.dart')) continue;
          final resolvedFile = _native(file.resolveSymbolicLinksSync());
          final unitRes = await session.getResolvedUnit(resolvedFile);
          if (unitRes is! ResolvedUnitResult) continue;

          final visitor = _ResolvedSemanticVisitor(
            file.path,
            unitRes.lineInfo,
            usedExtFunctions,
          );
          unitRes.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }

        final unusedExt = extFunctions.difference(usedExtFunctions);
        for (final u in unusedExt) {
          violations.add(
            'ndarray_extensions_bindings.dart — unused `@ffi.Native` binding `${u.name}`.',
          );
        }

        // 4. Verify operation_contracts_test.dart covers all BinaryOp & UnaryOp operations and branch coverage tool exists
        final contractsFile = File(
          '${pkgRoot.path}/test/meta/operation_contracts_test.dart',
        );
        final contractsSrc = contractsFile.readAsStringSync();
        const opAliasMap = <String, String>{
          'absolute': 'abs',
          'fabs': 'abs',
          'conjugate': 'conj',
          'arcsin': 'asin',
          'arccos': 'acos',
          'arctan': 'atan',
          'arcsinh': 'asinh',
          'arccosh': 'acosh',
          'arctanh': 'atanh',
          'arctan2': 'atan2',
          'degrees': 'rad2deg',
          'radians': 'deg2rad',
          'bitwiseNot': 'invert',
          'remainder': 'mod',
          'floatPower': 'power',
          'minimum': 'min',
          'maximum': 'max',
          'fmin': 'nanmin',
          'fmax': 'nanmax',
          'positive': 'add',
          'exp2': 'exp',
          'cbrt': 'sqrt',
          'signbit': 'sign',
          'spacing': 'abs',
        };
        for (final op in BinaryOp.values) {
          final fnName = opAliasMap[op.name] ?? op.name;
          if (!RegExp('\\b$fnName\\b').hasMatch(contractsSrc)) {
            violations.add(
              'BinaryOp.${op.name} (`$fnName`) is not exercised in test/meta/operation_contracts_test.dart',
            );
          }
        }
        for (final op in UnaryOp.values) {
          final fnName = opAliasMap[op.name] ?? op.name;
          if (!RegExp('\\b$fnName\\b').hasMatch(contractsSrc)) {
            violations.add(
              'UnaryOp.${op.name} (`$fnName`) is not exercised in test/meta/operation_contracts_test.dart',
            );
          }
        }
        if (!File(
          '${pkgRoot.path}/tool/check_branch_coverage.dart',
        ).existsSync()) {
          violations.add('Missing tool/check_branch_coverage.dart');
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Resolved semantic AST (`DartType` / `Element`) invariants violated:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'All workspace packages have identical standalone analysis_options.yaml ready for publishing',
      () {
        final canonicalFile = File('${pkgRoot.path}/analysis_options.yaml');
        expect(canonicalFile.existsSync(), isTrue);
        final canonicalContent = canonicalFile.readAsStringSync();
        expect(
          canonicalContent,
          contains('include: package:lints/recommended.yaml'),
        );
        expect(
          canonicalContent,
          isNot(contains('../../analysis_options.yaml')),
        );

        final pkgsDir = pkgRoot.parent;
        final mismatches = <String>[];
        for (final entity in pkgsDir.listSync().whereType<Directory>()) {
          final pubspec = File('${entity.path}/pubspec.yaml');
          if (!pubspec.existsSync()) continue;
          final optsFile = File('${entity.path}/analysis_options.yaml');
          if (!optsFile.existsSync()) {
            mismatches.add('${entity.path} is missing analysis_options.yaml');
            continue;
          }
          if (optsFile.readAsStringSync() != canonicalContent) {
            mismatches.add(
              '${optsFile.path} differs from canonical pkgs/ndarray/analysis_options.yaml',
            );
          }
        }
        expect(mismatches, isEmpty, reason: mismatches.join('\n'));
      },
    );

    test(
      'All NDArray mutator methods check isWriteable and same-dtype operators use _withSameDTypeOperand',
      () {
        final ndarrayFile = File('${pkgRoot.path}/lib/src/ndarray.dart');
        final parsed = parseFile(
          path: _native(ndarrayFile.path),
          featureSet: featureSet,
          throwIfDiagnostics: false,
        );

        const requiredWriteableMethods = {
          'fillUntyped',
          'setCellUntyped',
          'setCellRawUntyped',
          'setCellFlatUntyped',
          'setByMask',
          'setByMaskScalar',
          'setIndicesScalar',
          'setIndices',
          'sliceAssign',
          '[]=',
        };
        const sameDTypeOperators = {
          '+',
          '-',
          '*',
          '~/',
          '%',
          '&',
          '|',
          '^',
          '<<',
          '>>',
        };

        final visitor = _NDArrayMutatorAndOperatorVisitor(
          requiredWriteableMethods: requiredWriteableMethods,
          sameDTypeOperators: sameDTypeOperators,
        );
        parsed.unit.accept(visitor);

        final missing = requiredWriteableMethods.difference(
          visitor.foundMutators,
        );
        final violations = [...visitor.violations];
        if (missing.isNotEmpty) {
          violations.add('Could not find expected mutator methods: $missing');
        }

        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test(
      'All markdown Dart code blocks across doc/ and README.md parse cleanly and obey API invariants',
      () {
        final docFiles = [
          File('${pkgRoot.path}/README.md'),
          ...Directory(
            '${pkgRoot.path}/doc',
          ).listSync().whereType<File>().where((f) => f.path.endsWith('.md')),
        ];

        final dartBlockRegex = RegExp(r'```dart\s*\n([\s\S]*?)```');
        final bannedPatterns = [
          'NDArray<double>',
          'NDArray<int>',
          'NDArray<bool>',
          'Float64(',
          'Float32(',
          'Int32(',
          'Int64(',
          'view.detachToParentScope()',
          'view.detachFromScope()',
        ];

        final violations = <String>[];

        for (final file in docFiles) {
          if (!file.existsSync()) continue;
          final content = file.readAsStringSync();
          final matches = dartBlockRegex.allMatches(content);
          final relPath = _posix(file.path.substring(pkgRoot.path.length + 1));

          var blockIndex = 0;
          for (final match in matches) {
            blockIndex++;
            final code = match.group(1)!;

            // 1. Check for banned patterns
            for (final pattern in bannedPatterns) {
              if (code.contains(pattern)) {
                violations.add(
                  '$relPath block #$blockIndex contains banned pattern `$pattern`',
                );
              }
            }

            // 2. Syntax parse check
            final parseResult = parseString(
              content: code,
              featureSet: featureSet,
              throwIfDiagnostics: false,
            );
            if (parseResult.errors.isNotEmpty) {
              // If top-level statement diagnostics occur, try wrapping in a function
              final wrapped = 'void _snippet() {\n$code\n}';
              final wrappedResult = parseString(
                content: wrapped,
                featureSet: featureSet,
                throwIfDiagnostics: false,
              );
              if (wrappedResult.errors.isNotEmpty) {
                final errorsSummary = wrappedResult.errors
                    .map((e) {
                      final line = wrappedResult.lineInfo
                          .getLocation(e.offset)
                          .lineNumber;
                      return '    line $line: ${e.message}';
                    })
                    .join('\n');
                violations.add(
                  '$relPath block #$blockIndex syntax errors:\n$errorsSummary',
                );
              }
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Markdown documentation code block violations:\n${violations.join('\n')}',
        );
      },
    );

    test(
      'Package-level pubspec.yaml files do not contain buildMode: source or git/path runtime dependencies',
      () {
        final targetPkgs = ['ndarray', 'openblas', 'pocketfft'];
        final violations = <String>[];

        for (final pkgName in targetPkgs) {
          final pubspecFile = File('${pkgsDir.path}/$pkgName/pubspec.yaml');
          if (!pubspecFile.existsSync()) continue;
          final lines = pubspecFile.readAsLinesSync();

          var inRuntimeDeps = false;
          for (var i = 0; i < lines.length; i++) {
            final line = lines[i];
            final trimmed = line.trim();
            if (!line.startsWith(' ') &&
                !line.startsWith('\t') &&
                trimmed.isNotEmpty &&
                !trimmed.startsWith('#')) {
              inRuntimeDeps = trimmed == 'dependencies:';
            }
            if (line.contains('buildMode: source')) {
              violations.add(
                'pkgs/$pkgName/pubspec.yaml:${i + 1} — contains `buildMode: source` (belongs only in workspace root pubspec.yaml).',
              );
            }
            if (inRuntimeDeps &&
                (trimmed.startsWith('git:') || trimmed.startsWith('path:'))) {
              violations.add(
                'pkgs/$pkgName/pubspec.yaml:${i + 1} — contains `$trimmed` in runtime `dependencies:` (prohibited in publishable packages).',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Package-level pubspec.yaml must not contain buildMode: source or git/path runtime dependencies:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'AST scan of error messages in pkgs/ndarray/lib/ has no broken empty interpolation patterns',
      () {
        final violations = <String>[];
        for (final file in libFiles) {
          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _ErrorMessageInterpolationVisitor(
            file.path,
            parsed.lineInfo,
          );
          parsed.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Found error instantiation with missing variable interpolation:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'All first-party source files carry the Google LLC Apache 2.0 header, LICENSE files use canonical template, and no legacy repo references remain',
      () {
        final workspaceRoot = pkgsDir.parent;
        final violations = <String>[];

        // 1. Check all LICENSE files in workspace root and pkgs/*
        final licenseFiles = <File>[
          File('${workspaceRoot.path}/LICENSE'),
          for (final dir in pkgsDir.listSync().whereType<Directory>())
            if (File('${dir.path}/LICENSE').existsSync())
              File('${dir.path}/LICENSE'),
        ];
        for (final lf in licenseFiles) {
          final text = lf.readAsStringSync();
          if (!text.contains('Copyright [yyyy] [name of copyright owner]')) {
            violations.add(
              '${lf.path} — missing canonical Apache 2.0 boilerplate (`Copyright [yyyy] [name of copyright owner]`).',
            );
          }
          if (text.contains('Sigurd Meldgaard')) {
            violations.add(
              '${lf.path} — contains legacy copyright holder name.',
            );
          }
        }

        // 2. Check all first-party .dart, .c, .cpp, .h files across pkgs/ and tool/
        const excludedSubpaths = [
          '/third_party/',
          '/hook/include/',
          '/hook/pocketfft_hdronly.h',
          '/lib/src/miniaudio.h',
          '/.dart_tool/',
          '/build/',
        ];
        bool isExcluded(String posixPath) =>
            excludedSubpaths.any((sub) => posixPath.contains(sub));

        final dirsToScan = [pkgsDir, Directory('${workspaceRoot.path}/tool')];
        for (final root in dirsToScan) {
          if (!root.existsSync()) continue;
          for (final entity in root.listSync(recursive: true)) {
            if (entity is! File) continue;
            final posixPath = _posix(entity.path);
            if (isExcluded(posixPath)) continue;

            if (posixPath.endsWith('.dart') ||
                posixPath.endsWith('.c') ||
                posixPath.endsWith('.cpp') ||
                posixPath.endsWith('.h')) {
              final content = entity.readAsStringSync();
              if (!content.contains('// Copyright 2026 Google LLC')) {
                violations.add(
                  '$posixPath — missing `// Copyright 2026 Google LLC` Apache 2.0 header.',
                );
              }
              if (!posixPath.endsWith('codebase_invariants_test.dart') &&
                  (content.contains('sigurdm/scientific_dart') ||
                      content.contains('math_workspace'))) {
                violations.add(
                  '$posixPath — contains legacy repository/workspace reference (`sigurdm/scientific_dart` or `math_workspace`).',
                );
              }
            } else if (posixPath.endsWith('.yaml') ||
                posixPath.endsWith('.yml') ||
                posixPath.endsWith('.md')) {
              final content = entity.readAsStringSync();
              if (content.contains('sigurdm/scientific_dart') ||
                  content.contains('math_workspace')) {
                violations.add(
                  '$posixPath — contains legacy repository/workspace reference (`sigurdm/scientific_dart` or `math_workspace`).',
                );
              }
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'License header or repository URL invariant violations:\n'
              '${violations.join('\n')}',
        );
      },
    );
  });

  group('64-Bit Size, Stride, and Index Invariants (R1–R4)', () {
    test(
      'Native C++ headers use int64_t (never 32-bit int*) for shape, stride, and index arrays',
      () {
        final headers = [
          File('${pkgRoot.path}/hook/custom_ufuncs.h'),
          File('${pkgRoot.path}/hook/custom_sorting.h'),
          File('${pkgRoot.path}/hook/custom_indexing.h'),
        ];
        final forbiddenShapeStridePtr = RegExp(
          r'\bconst\s+int\s*\*\s*(shape|strides|[a-z]+_strides)\b',
        );
        final forbiddenIndexPtr = RegExp(
          r'\bint\s*\*\s*(out_indices|indices)\b',
        );
        final violations = <String>[];

        for (final header in headers) {
          expect(header.existsSync(), isTrue, reason: 'Missing ${header.path}');
          final content = header.readAsStringSync();
          if (!content.contains('int64_t')) {
            violations.add('${header.path}: does not use int64_t');
          }
          for (final match in forbiddenShapeStridePtr.allMatches(content)) {
            violations.add(
              '${header.path}: found 32-bit shape/stride pointer "${match.group(0)}"',
            );
          }
          for (final match in forbiddenIndexPtr.allMatches(content)) {
            violations.add(
              '${header.path}: found 32-bit index pointer "${match.group(0)}"',
            );
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Found 32-bit int* shape/stride/index declarations in C++ headers:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'Dart FFI bindings use ffi.Int64 and ffi.Pointer<ffi.Int64> for shapes, strides, and index buffers',
      () {
        final bindingsFiles = [
          File('${pkgRoot.path}/lib/src/ndarray_bindings.dart'),
          File('${pkgRoot.path}/lib/src/ndarray_extensions_bindings.dart'),
        ];
        final forbiddenShapeStrideOrIndexParam = RegExp(
          r'ffi\.Pointer<ffi\.Int(32)?>\s+(shape|strides|[a-zA-Z]+Strides|[a-z]+_strides|out_indices|indices|start_coords|directions|match_coords)\b',
        );
        final violations = <String>[];

        for (final file in bindingsFiles) {
          final lines = file.readAsLinesSync();
          for (var i = 0; i < lines.length; i++) {
            final line = lines[i];
            if (forbiddenShapeStrideOrIndexParam.hasMatch(line)) {
              violations.add(
                '${_posix(file.path)}:${i + 1}: 32-bit shape/stride/index pointer parameter: ${line.trim()}',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'FFI bindings must use ffi.Pointer<ffi.Int64> for shape, stride, and index arrays:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'ScratchArena helpers return ffi.Pointer<ffi.Int64> for strided metadata and integer copies',
      () {
        final arenaFile = File('${pkgRoot.path}/lib/src/scratch_arena.dart');
        final content = arenaFile.readAsStringSync();
        expect(
          content,
          contains('static ffi.Pointer<ffi.Int64> getStridedBuffer('),
        );
        expect(content, contains('static ffi.Pointer<ffi.Int64> copyInts('));
        expect(content, contains('static ffi.Pointer<ffi.Int64> copyInt64s('));
      },
    );

    test(
      'Public index-producing and count-producing operations return NDArray<Int64>',
      () {
        final sortingContent = File(
          '${pkgRoot.path}/lib/src/operations/sorting.dart',
        ).readAsStringSync();
        expect(sortingContent, matches(RegExp(r'NDArray<Int64>\s+argsort\b')));
        expect(
          sortingContent,
          matches(RegExp(r'NDArray<Int64>\s+argpartition\b')),
        );
        expect(
          sortingContent,
          matches(RegExp(r'NDArray<Int64>\s+searchsorted\b')),
        );
        expect(sortingContent, matches(RegExp(r'NDArray<Int64>\s+argmax\b')));
        expect(sortingContent, matches(RegExp(r'NDArray<Int64>\s+argmin\b')));
        expect(
          sortingContent,
          matches(RegExp(r'List<NDArray<Int64>>\s+nonzero\b')),
        );
        expect(sortingContent, matches(RegExp(r'NDArray<Int64>\s+argwhere\b')));
        expect(
          sortingContent,
          matches(RegExp(r'NDArray<Int64>\s+count_nonzero\b')),
        );
        expect(
          sortingContent,
          matches(RegExp(r'NDArray<Int64>\s+flatnonzero\b')),
        );

        final binningContent = File(
          '${pkgRoot.path}/lib/src/operations/binning.dart',
        ).readAsStringSync();
        expect(binningContent, matches(RegExp(r'NDArray<Int64>\s+digitize\b')));

        final indexingContent = File(
          '${pkgRoot.path}/lib/src/operations/indexing.dart',
        ).readAsStringSync();
        expect(
          indexingContent,
          matches(RegExp(r'List<NDArray<Int64>>\s+unravel_index\b')),
        );
        expect(
          indexingContent,
          matches(RegExp(r'NDArray<Int64>\s+ravel_multi_index\b')),
        );
        expect(
          indexingContent,
          matches(RegExp(r'List<NDArray<Int64>>\s+diag_indices\b')),
        );
        expect(
          indexingContent,
          matches(RegExp(r'List<NDArray<Int64>>\s+diag_indices_from\b')),
        );
        expect(
          indexingContent,
          matches(
            RegExp(
              r'\(\{NDArray<Int64>\s+row,\s*NDArray<Int64>\s+col\}\)\s+tril_indices\b',
            ),
          ),
        );
        expect(
          indexingContent,
          matches(
            RegExp(
              r'\(\{NDArray<Int64>\s+row,\s*NDArray<Int64>\s+col\}\)\s+triu_indices\b',
            ),
          ),
        );
        expect(
          indexingContent,
          matches(
            RegExp(
              r'\(\{NDArray<Int64>\s+row,\s*NDArray<Int64>\s+col\}\)\s+mask_indices\b',
            ),
          ),
        );

        final setOpsContent = File(
          '${pkgRoot.path}/lib/src/operations/set_operations.dart',
        ).readAsStringSync();
        expect(
          setOpsContent,
          matches(RegExp(r'NDArray<T>\s+unique<T\s+extends\s+DTypeTag>')),
        );
        expect(
          setOpsContent,
          matches(
            RegExp(
              r'\(\{NDArray<T>\s+values,\s*NDArray<Int64>\s+index\}\)\s+uniqueWithIndex<T\s+extends\s+DTypeTag>',
            ),
          ),
        );
        expect(
          setOpsContent,
          matches(
            RegExp(
              r'\(\{NDArray<T>\s+values,\s*NDArray<Int64>\s+inverse\}\)\s+uniqueWithInverse<T\s+extends\s+DTypeTag>',
            ),
          ),
        );
        expect(
          setOpsContent,
          matches(
            RegExp(
              r'\(\{NDArray<T>\s+values,\s*NDArray<Int64>\s+counts\}\)\s+uniqueWithCounts<T\s+extends\s+DTypeTag>',
            ),
          ),
        );
        expect(
          setOpsContent,
          matches(
            RegExp(
              r'\(\{\s*NDArray<T>\s+values,\s*NDArray<Int64>\s+index,\s*NDArray<Int64>\s+inverse,\s*NDArray<Int64>\s+counts,?\s*\}\)\s+uniqueAll<T\s+extends\s+DTypeTag>',
            ),
          ),
        );
      },
    );

    test(
      'Core NDArray, broadcasting, slicing, and NPY/NPZ I/O have no 32-bit 0x7fffffff ceilings',
      () {
        final hexCeilingPattern = RegExp(
          r'\b0x7fffffff\b',
          caseSensitive: false,
        );
        final decCeilingPattern = RegExp(r'\b2147483647\b');
        final coreFiles = [
          File('${pkgRoot.path}/lib/src/ndarray.dart'),
          File('${pkgRoot.path}/lib/src/operations/broadcasting.dart'),
          File('${pkgRoot.path}/lib/src/operations/spacers.dart'),
          File('${pkgRoot.path}/lib/src/operations/io.dart'),
        ];
        final violations = <String>[];
        for (final file in coreFiles) {
          final content = file.readAsStringSync();
          if (hexCeilingPattern.hasMatch(content)) {
            violations.add(
              '${_posix(file.path)} still contains a 32-bit 0x7fffffff ceiling.',
            );
          }
          if (!file.path.endsWith('ndarray.dart') &&
              decCeilingPattern.hasMatch(content)) {
            violations.add(
              '${_posix(file.path)} still contains a 32-bit 2147483647 ceiling.',
            );
          }
        }
        final ndarrayContent = File(
          '${pkgRoot.path}/lib/src/ndarray.dart',
        ).readAsStringSync();
        expect(
          ndarrayContent,
          contains('totalSize > 0x7fffffffffffffff ~/ dim'),
          reason:
              'NDArray._computeCheckedTotalSize must check 64-bit signed multiplication overflow.',
        );
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test('No public typedef in lib/ shadows dart:ffi type names', () {
      final bannedNames = {
        'Float',
        'Double',
        'Int8',
        'Int16',
        'Int32',
        'Int64',
        'Uint8',
        'Uint16',
        'Uint32',
        'Uint64',
        'Size',
        'IntPtr',
        'UintPtr',
      };

      final violations = <String>[];
      for (final file in libFiles) {
        final parsed = parseFile(
          path: _native(file.path),
          featureSet: featureSet,
          throwIfDiagnostics: false,
        );
        for (final decl in parsed.unit.declarations) {
          if (decl is GenericTypeAlias) {
            final name = decl.name.lexeme;
            if (bannedNames.contains(name) && !name.startsWith('_')) {
              final line = parsed.lineInfo.getLocation(decl.offset).lineNumber;
              violations.add(
                '${_posix(file.path)}:$line — public typedef `$name` shadows dart:ffi type.',
              );
            }
          } else if (decl is FunctionTypeAlias) {
            final name = decl.name.lexeme;
            if (bannedNames.contains(name) && !name.startsWith('_')) {
              final line = parsed.lineInfo.getLocation(decl.offset).lineNumber;
              violations.add(
                '${_posix(file.path)}:$line — public typedef `$name` shadows dart:ffi type.',
              );
            }
          }
        }
      }

      expect(
        violations,
        isEmpty,
        reason:
            'Public typedefs in lib/ must not shadow dart:ffi type names:\n${violations.join('\n')}',
      );
    });

    test(
      'Complex, IndexSpec, and Index value/spec types have const generative constructors',
      () {
        final targetClasses = {'Complex', 'Index', 'Selector'};
        final found = <String>{};
        final violations = <String>[];

        for (final file in libFiles) {
          final parsed = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          for (final decl in parsed.unit.declarations) {
            if (decl is ClassDeclaration &&
                targetClasses.contains(decl.namePart.typeName.lexeme)) {
              final className = decl.namePart.typeName.lexeme;
              found.add(className);
              final constConstructors = decl.body.members
                  .whereType<ConstructorDeclaration>()
                  .where(
                    (c) => c.constKeyword != null && c.factoryKeyword == null,
                  )
                  .toList();
              if (constConstructors.isEmpty) {
                violations.add(
                  '$className is missing a const generative constructor.',
                );
              }
            }
          }
        }

        expect(found, containsAll(targetClasses));
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test(
      'Defensive collection copying: Indices, CoordinateSpacing, TensordotAxes, and BroadcastResult wrap list fields with List.unmodifiable',
      () {
        final targetClasses = {
          'Indices': 'values',
          'CoordinateSpacing': 'values',
          'TensordotAxes': 'explicitAxesA',
          'BroadcastResult': 'shape',
        };
        final foundClasses = <String>{};

        for (final file in libFiles) {
          final content = file.readAsStringSync();
          for (final entry in targetClasses.entries) {
            final cls = entry.key;
            if (content.contains('class $cls')) {
              foundClasses.add(cls);
              expect(
                content.contains('List') && content.contains('.unmodifiable('),
                isTrue,
                reason:
                    '$cls in ${file.path} must defensively wrap its list fields using List.unmodifiable',
              );
            }
          }
        }

        expect(foundClasses, containsAll(targetClasses.keys));
      },
    );

    test('Enum value documentation & {@example} tag hygiene across lib/', () {
      final violations = <String>[];
      final exampleDir = Directory('${pkgRoot.path}/example');
      expect(exampleDir.existsSync(), isTrue);

      // 1. Assert all public enum values in transitively exported public enums have /// comments
      final entrypoint = File('${libDir.path}/ndarray.dart');
      final visited = <String>{};
      final exportedFiles = <(File, Set<String>?, Set<String>?)>[];

      void collectExports(
        File file, {
        Set<String>? showNames,
        Set<String>? hideNames,
      }) {
        final canonical = file.resolveSymbolicLinksSync();
        if (!visited.add('$canonical|$showNames|$hideNames')) return;
        exportedFiles.add((file, showNames, hideNames));

        final parsed = parseFile(
          path: _native(file.path),
          featureSet: featureSet,
          throwIfDiagnostics: false,
        );
        for (final directive in parsed.unit.directives) {
          if (directive is ExportDirective) {
            final uriStr = directive.uri.stringValue;
            if (uriStr == null || uriStr.startsWith('package:')) continue;
            final resolved = File('${file.parent.path}/$uriStr');
            if (!resolved.existsSync()) continue;

            Set<String>? childShow = showNames;
            final childHide = <String>{...?hideNames};
            for (final combinator in directive.combinators) {
              if (combinator is ShowCombinator) {
                final names = combinator.shownNames.map((n) => n.name).toSet();
                childShow = childShow == null
                    ? names
                    : childShow.intersection(names);
              } else if (combinator is HideCombinator) {
                childHide.addAll(combinator.hiddenNames.map((n) => n.name));
              }
            }
            collectExports(
              resolved,
              showNames: childShow,
              hideNames: childHide,
            );
          }
        }
      }

      collectExports(entrypoint);

      for (final (file, showNames, hideNames) in exportedFiles) {
        final parsed = parseFile(
          path: _native(file.path),
          featureSet: featureSet,
          throwIfDiagnostics: false,
        );
        for (final decl in parsed.unit.declarations) {
          if (decl is EnumDeclaration) {
            final enumName = decl.namePart.typeName.lexeme;
            if (enumName.startsWith('_')) continue;
            if (showNames != null && !showNames.contains(enumName)) continue;
            if (hideNames != null && hideNames.contains(enumName)) continue;

            for (final constDecl in decl.body.constants) {
              final constName = constDecl.name.lexeme;
              if (constDecl.documentationComment == null) {
                final line = parsed.lineInfo
                    .getLocation(constDecl.offset)
                    .lineNumber;
                violations.add(
                  '${_posix(file.path)}:$line — enum value `$enumName.$constName` is missing /// dartdoc',
                );
              }
            }
          }
        }
      }

      // 2. Assert {@example ...} tag hygiene across lib/ and zero inline ```dart blocks in P2-6 modules
      const p26Modules = <String>{
        'binning.dart',
        'broadcasting.dart',
        'calculus.dart',
        'dsp.dart',
        'indexing.dart',
        'repeating_tiling.dart',
        'set_operations.dart',
      };
      for (final file in libFiles) {
        final posixPath = _posix(file.path);
        final baseName = file.uri.pathSegments.last;
        final isTopLevelOpModule =
            RegExp(r'/src/operations/[^/]+\.dart$').hasMatch(posixPath) &&
            !const {
              'helpers.dart',
              'math.dart',
              'polynomial.dart',
              'custom_checks.dart',
              'native_pointer.dart',
            }.contains(baseName);
        final lines = file.readAsLinesSync();
        var hasExampleTag = false;
        for (var i = 0; i < lines.length; i++) {
          final line = lines[i];
          if (p26Modules.contains(baseName) &&
              RegExp(r'^\s*///\s*```dart\b').hasMatch(line)) {
            violations.add(
              '$posixPath:${i + 1} — inline ```` ```dart ```` code block in dartdoc; use `{@example /example/... lang=dart}` instead.',
            );
          }
          if (line.contains('{@example')) {
            hasExampleTag = true;
            // Must be on its own line
            if (!RegExp(
              r'^\s*///\s*\{@example\s+[^\n]+?\}\s*$',
            ).hasMatch(line)) {
              violations.add(
                '$posixPath:${i + 1} — `{@example}` tag must appear on its own line: `$line`',
              );
            }
            // Extract target path
            final m = RegExp(r'\{@example\s+([^\s}]+)').firstMatch(line);
            if (m != null) {
              final rawPath = m.group(1)!;
              if (rawPath.contains('pkgs/ndarray/example/')) {
                violations.add(
                  '$posixPath:${i + 1} — `{@example}` must not contain `pkgs/ndarray/example/`: `$rawPath`',
                );
              }
              final cleanedPath = rawPath
                  .replaceFirst(RegExp(r'^/+'), '')
                  .split('#')
                  .first;
              final resolved = File('${pkgRoot.path}/$cleanedPath');
              if (!resolved.existsSync()) {
                violations.add(
                  '$posixPath:${i + 1} — `{@example}` references non-existent file: `$rawPath` -> `$cleanedPath`',
                );
              }
            }
          }
        }
        if (isTopLevelOpModule && !hasExampleTag) {
          violations.add(
            '$posixPath — operations module is missing `{@example /example/... lang=dart}` tags.',
          );
        }
      }

      expect(
        violations,
        isEmpty,
        reason:
            'Enum value documentation & {@example} tag hygiene violations:\n${violations.join('\n')}',
      );
    });

    test('Native C++ SIMD & ufunc dispatch invariants (P1-1, P1-2, P1-3)', () {
      // P1-1: ndarray_unique in custom_sorting.cpp dispatches all 10 integer/float DTypes to unique_*_fast
      final sortingCpp = File(
        '${pkgRoot.path}/hook/custom_sorting.cpp',
      ).readAsStringSync();
      final uniqueIdx = sortingCpp.indexOf('int64_t ndarray_unique(');
      expect(uniqueIdx, greaterThan(0));
      final uniqueBody = sortingCpp.substring(uniqueIdx);
      for (final fastFn in [
        'unique_double_fast',
        'unique_float_fast',
        'unique_int64_fast',
        'unique_int32_fast',
        'unique_int16_fast',
        'unique_int8_fast',
        'unique_uint64_fast',
        'unique_uint32_fast',
        'unique_uint16_fast',
        'unique_uint8_fast',
      ]) {
        expect(
          uniqueBody,
          contains(fastFn),
          reason: 'ndarray_unique must dispatch to $fastFn',
        );
      }
      expect(
        uniqueBody,
        isNot(contains('unique_scalar_fast<')),
        reason:
            'ndarray_unique must not fall back directly to unique_scalar_fast<T> for integer/float DTypes',
      );

      // P1-2 & P1-3: ufunc_methods.dart dispatches minimum/maximum/fmin/fmax to v_binary_minmax / s_binary_minmax
      final ufuncMethods = File(
        '${pkgRoot.path}/lib/src/operations/math/ufunc_methods.dart',
      ).readAsStringSync();
      expect(ufuncMethods, contains('v_binary_minmax('));
      expect(ufuncMethods, contains('s_binary_minmax('));

      // P1-3: outerUfunc delegates directly to binaryUfunc without manual ffi.Pointer loops
      final outerStart = ufuncMethods.indexOf('NDArray<T> outerUfunc<');
      final atStart = ufuncMethods.indexOf('void atUfunc<');
      expect(outerStart, greaterThan(0));
      expect(atStart, greaterThan(outerStart));
      final outerBody = ufuncMethods.substring(outerStart, atStart);
      expect(
        outerBody,
        isNot(contains('.pointer.cast<')),
        reason:
            'outerUfunc must delegate to binaryUfunc (native C strided kernels) rather than manual Dart pointer loops',
      );
    });
  });
}

class _NDArrayMutatorAndOperatorVisitor extends RecursiveAstVisitor<void> {
  final Set<String> requiredWriteableMethods;
  final Set<String> sameDTypeOperators;
  final Set<String> foundMutators = {};
  final List<String> violations = [];

  _NDArrayMutatorAndOperatorVisitor({
    required this.requiredWriteableMethods,
    required this.sameDTypeOperators,
  });

  @override
  void visitClassDeclaration(ClassDeclaration node) {
    if (node.namePart.typeName.lexeme == 'NDArray') {
      super.visitClassDeclaration(node);
    }
  }

  @override
  void visitMethodDeclaration(MethodDeclaration member) {
    final name = member.name.lexeme;
    final bodySrc = member.body.toSource();
    if (requiredWriteableMethods.contains(name)) {
      foundMutators.add(name);
      if (!bodySrc.contains('!isWriteable')) {
        violations.add('NDArray.$name is missing `if (!isWriteable)` guard.');
      }
    }
    if (member.isOperator &&
        sameDTypeOperators.contains(name) &&
        member.parameters?.parameters.length == 1) {
      if (!bodySrc.contains('_withSameDTypeOperand')) {
        violations.add(
          'NDArray.operator $name must dispatch via `_withSameDTypeOperand` (never raw `as NDArray<T>`).',
        );
      }
    }
  }
}

class _ErrorMessageInterpolationVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final List<String> violations = [];

  static const _errorClasses = {
    'ArgumentError',
    'RangeError',
    'StateError',
    'UnsupportedError',
    'FormatException',
  };

  static final _brokenPattern = RegExp(
    r'(?:'
    r'was rank \)|'
    r'of rank \.|'
    r'of rank \x27|'
    r'of rank \x22|'
    r'\(was \)'
    r')',
  );

  _ErrorMessageInterpolationVisitor(this.filePath, this.lineInfo);

  @override
  void visitThrowExpression(ThrowExpression node) {
    _checkExpression(node.expression);
    super.visitThrowExpression(node);
  }

  void _checkExpression(Expression expr) {
    if (expr is InstanceCreationExpression) {
      final typeName = expr.constructorName.type.name.lexeme;
      if (_errorClasses.contains(typeName)) {
        for (final arg in expr.argumentList.arguments) {
          _inspectArgument(arg);
        }
      }
    } else if (expr is MethodInvocation) {
      final target = expr.target?.toSource();
      if (target != null && _errorClasses.contains(target)) {
        for (final arg in expr.argumentList.arguments) {
          _inspectArgument(arg);
        }
      }
    }
  }

  void _inspectArgument(AstNode arg) {
    final finder = _BrokenStringVisitor(filePath, lineInfo, _brokenPattern);
    arg.accept(finder);
    violations.addAll(finder.violations);
  }
}

class _BrokenStringVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final RegExp pattern;
  final List<String> violations = [];

  _BrokenStringVisitor(this.filePath, this.lineInfo, this.pattern);

  @override
  void visitSimpleStringLiteral(SimpleStringLiteral node) {
    if (pattern.hasMatch(node.value)) {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line — broken string literal: "${node.value}"',
      );
    }
    super.visitSimpleStringLiteral(node);
  }

  @override
  void visitStringInterpolation(StringInterpolation node) {
    for (final element in node.elements) {
      if (element is InterpolationString && pattern.hasMatch(element.value)) {
        final line = lineInfo.getLocation(element.offset).lineNumber;
        violations.add(
          '$filePath:$line — broken interpolation string element: "${element.value}" in "${node.toSource()}"',
        );
      }
    }
    super.visitStringInterpolation(node);
  }
}

class _ResolvedSemanticVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final Set<ExecutableElement> usedExtFunctions;
  final List<String> violations = [];

  _ResolvedSemanticVisitor(this.filePath, this.lineInfo, this.usedExtFunctions);

  bool _isNDArrayType(DartType? type) =>
      type is InterfaceType && type.element.name == 'NDArray';

  @override
  void visitSimpleIdentifier(SimpleIdentifier node) {
    final el = node.element;
    if (el is ExecutableElement) {
      usedExtFunctions.add(el);
    }
    super.visitSimpleIdentifier(node);
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final isOperations = _posix(filePath).contains('/src/operations/');
    final targetType = node.realTarget?.staticType;

    // Receiver-type-aware ban on NDArray.toList() in lib/src/operations/
    if (isOperations &&
        _isNDArrayType(targetType) &&
        node.methodName.name == 'toList') {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line — `NDArray.toList()` called in operations (`${node.toSource()}`); use views, `NDIter`, or FFI pointers instead.',
      );
    }

    // Resolved DartType equality between ScratchArena.allocate<T>(...) and sizeOf<U>()
    if (node.target?.toSource() == 'ScratchArena' &&
        node.methodName.name == 'allocate') {
      final allocType = node.typeArgumentTypes?.firstOrNull;
      final arg = node.argumentList.arguments.firstOrNull;
      if (allocType != null && arg != null) {
        final sizeOfFinder = _FindSizeOfTypeVisitor();
        arg.accept(sizeOfFinder);
        for (final sizeType in sizeOfFinder.sizeOfTypes) {
          if (allocType != sizeType) {
            final line = lineInfo.getLocation(node.offset).lineNumber;
            violations.add(
              '$filePath:$line — `ScratchArena.allocate<${allocType.getDisplayString()}>` byte count uses mismatched `sizeOf<${sizeType.getDisplayString()}>()`.',
            );
          }
        }
      }
    }

    super.visitMethodInvocation(node);
  }

  @override
  void visitPropertyAccess(PropertyAccess node) {
    if (_posix(filePath).contains('/src/operations/') &&
        _isNDArrayType(node.realTarget.staticType) &&
        node.propertyName.name == 'data') {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line — `@internal` `NDArray.data` accessed in operations (`${node.toSource()}`).',
      );
    }
    super.visitPropertyAccess(node);
  }

  @override
  void visitPrefixedIdentifier(PrefixedIdentifier node) {
    if (_posix(filePath).contains('/src/operations/') &&
        _isNDArrayType(node.prefix.staticType) &&
        node.identifier.name == 'data') {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line — `@internal` `NDArray.data` accessed in operations (`${node.toSource()}`).',
      );
    }
    super.visitPrefixedIdentifier(node);
  }
}

class _FindSizeOfTypeVisitor extends RecursiveAstVisitor<void> {
  final List<DartType> sizeOfTypes = [];

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name == 'sizeOf') {
      final t = node.typeArgumentTypes?.firstOrNull;
      if (t != null) sizeOfTypes.add(t);
    }
    super.visitMethodInvocation(node);
  }

  @override
  void visitFunctionExpressionInvocation(FunctionExpressionInvocation node) {
    if (node.function.toSource().endsWith('sizeOf')) {
      final t = node.typeArgumentTypes?.firstOrNull;
      if (t != null) sizeOfTypes.add(t);
    }
    super.visitFunctionExpressionInvocation(node);
  }
}

class _ScratchArenaVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final List<String> violations = [];

  _ScratchArenaVisitor(this.filePath, this.lineInfo);

  @override
  void visitVariableDeclarationStatement(VariableDeclarationStatement node) {
    for (final variable in node.variables.variables) {
      final init = variable.initializer;
      if (init != null && init.toSource() == 'ScratchArena.marker') {
        final markerName = variable.name.lexeme;
        final parent = node.parent;
        final List<Statement>? statements = switch (parent) {
          Block(:final statements) => statements,
          SwitchPatternCase(:final statements) => statements,
          SwitchCase(:final statements) => statements,
          SwitchDefault(:final statements) => statements,
          _ => null,
        };
        if (statements != null) {
          final idx = statements.indexOf(node);
          final hasTryFinallyReset = statements
              .skip(idx + 1)
              .whereType<TryStatement>()
              .any((tryStmt) {
                final finallySrc = tryStmt.finallyBlock?.toSource() ?? '';
                return finallySrc.contains('ScratchArena.reset($markerName)');
              });
          if (!hasTryFinallyReset) {
            final line = lineInfo.getLocation(node.offset).lineNumber;
            violations.add(
              '$filePath:$line — `ScratchArena.marker` stored in `$markerName` is not paired with `try { ... } finally { ScratchArena.reset($markerName); }`.',
            );
          }
        } else {
          final line = lineInfo.getLocation(node.offset).lineNumber;
          violations.add(
            '$filePath:$line — `ScratchArena.marker` must be declared in a Block or SwitchCase before a TryStatement.',
          );
        }
      }
    }
    super.visitVariableDeclarationStatement(node);
  }
}

class _UncheckedNativeCallVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final Set<String> intReturningNativeFunctions;
  final List<String> violations = [];

  _UncheckedNativeCallVisitor(
    this.filePath,
    this.lineInfo,
    this.intReturningNativeFunctions,
  );

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final name = node.methodName.name;
    if (intReturningNativeFunctions.contains(name)) {
      if (node.parent is ExpressionStatement) {
        final line = lineInfo.getLocation(node.offset).lineNumber;
        violations.add(
          '$filePath:$line — return value of `$name(...)` is ignored.',
        );
      }
    }
    super.visitMethodInvocation(node);
  }
}

class _FinalizerExternalSizeVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final List<String> violations = [];

  _FinalizerExternalSizeVisitor(this.filePath, this.lineInfo);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name == 'attach') {
      for (final arg in node.argumentList.arguments) {
        if (arg.toSource().startsWith('externalSize:')) {
          final line = lineInfo.getLocation(arg.offset).lineNumber;
          violations.add(
            '$filePath:$line — `externalSize:` passed to `.attach(...)`.',
          );
        }
      }
    }
    super.visitMethodInvocation(node);
  }
}
