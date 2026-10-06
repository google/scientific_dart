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

/// Represents a single notebook cell input for Wasm compilation or analysis.
final class WasmNotebookCell {
  /// Unique identifier of the cell in the notebook DOM.
  final String id;

  /// Raw source text of the cell.
  final String code;

  /// Cell type (`'code'` or `'markdown'`).
  final String type;

  /// Constructs a [WasmNotebookCell].
  const WasmNotebookCell({
    required this.id,
    required this.code,
    this.type = 'code',
  });

  /// Constructs a [WasmNotebookCell] from a JSON map.
  factory WasmNotebookCell.fromJson(Map<String, Object?> json) =>
      WasmNotebookCell(
        id: (json['id'] as String?) ?? '',
        code: (json['code'] as String?) ?? '',
        type: (json['type'] as String?) ?? 'code',
      );

  /// Converts this cell to a JSON map.
  Map<String, Object?> toJson() => {'id': id, 'code': code, 'type': type};
}

/// Result of bundling a sequence of notebook cells into a single Dart entrypoint.
final class WasmBundledProgram {
  /// Complete Dart source code for the combined entrypoint.
  final String mainDartSource;

  /// Ordered list of code cell IDs included in the bundle.
  final List<String> includedCellIds;

  /// Ordered list of user-declared variable names across the bundled cells.
  final List<String> declaredVariables;

  /// Constructs a [WasmBundledProgram].
  const WasmBundledProgram({
    required this.mainDartSource,
    required this.includedCellIds,
    required this.declaredVariables,
  });
}

/// Result of bundling notebook cells for LSP completion/hover analysis.
final class WasmAnalysisBundle {
  /// Combined Dart source code suitable for analysis.
  final String source;

  /// Mapped 0-based character offset in [source] corresponding to the cursor in the active cell.
  final int mappedOffset;

  /// Constructs a [WasmAnalysisBundle].
  const WasmAnalysisBundle({required this.source, required this.mappedOffset});
}

/// Bundles multi-cell notebook sessions into a single compilable Dart program
/// for serverless Wasm execution (re-run-from-top state model) and LSP analysis.
final class WasmCellBundler {
  const WasmCellBundler._();

  static const List<String> _defaultImports = [
    "import 'dart:async';",
    "import 'dart:convert';",
    "import 'dart:js_interop';",
    "import 'dart:math' as math;",
    "import 'dart:typed_data';",
    "import 'package:ndarray/ndarray.dart';",
    "import 'package:notebook/src/notebook_widgets.dart';",
    "import 'package:resource_scope/resource_scope.dart';",
  ];

  static const Set<String> _statementKeywords = {
    'assert',
    'await',
    'break',
    'case',
    'catch',
    'continue',
    'default',
    'do',
    'else',
    'finally',
    'for',
    'if',
    'rethrow',
    'return',
    'switch',
    'throw',
    'try',
    'while',
    'yield',
  };

  /// Bundles [cells] (up to and including [targetCellId], or all code cells if
  /// [targetCellId] is `null`) into a single compilable Dart program.
  static WasmBundledProgram bundleCells(
    List<WasmNotebookCell> cells, {
    String? targetCellId,
  }) {
    final selectedCells = <WasmNotebookCell>[];
    for (final cell in cells) {
      if (cell.type == 'code') {
        selectedCells.add(cell);
      }
      if (targetCellId != null && cell.id == targetCellId) {
        break;
      }
    }

    final userImports = <String>[];
    final seenImports = <String>{..._defaultImports};
    var usesGpuArray = selectedCells.any(
      (c) => _gpuUsageRegex.hasMatch(c.code),
    );
    if (usesGpuArray) {
      seenImports.add(_gpuCoreImport);
      userImports.add(_gpuCoreImport);
    }
    final topLevelDeclarations = <String, String>{};
    final declaredVariables = <String>[];
    final seenVariables = <String>{};
    final varDeclCounts = <String, int>{};
    final includedCellIds = <String>[];

    // Pre-pass: count variable declarations across cells so uniquely declared
    // variables keep their exact Dart static type inference (e.g. NDArray<Float64>)
    // in the shared main() scope, while variables re-declared across multiple
    // cells fall back to a shared `dynamic` slot.
    for (final cell in selectedCells) {
      final rawTrimmed = cell.code.trim();
      if (rawTrimmed.isEmpty) continue;
      if (RegExp(
        r'^(?:%)?(?:pub\s+add|add)\s+([\w\d_\-]+)\s*;?$',
      ).hasMatch(rawTrimmed)) {
        continue;
      }
      final extracted = _extractImportsAndBody(cell.code);
      for (final imp in extracted.imports) {
        if (imp.contains('package:gpuarray/')) {
          usesGpuArray = true;
          if (seenImports.add(_gpuCoreImport)) {
            userImports.add(_gpuCoreImport);
          }
        }
      }
      for (final item in _splitTopLevelItems(extracted.body)) {
        final kind = _classifyItem(item);
        if (kind.type == _ItemType.variableDeclaration) {
          for (final v in kind.variables) {
            varDeclCounts[v.name] = (varDeclCounts[v.name] ?? 0) + 1;
          }
        }
      }
    }

    final redeclaredVars = <String>{
      for (final entry in varDeclCounts.entries)
        if (entry.value > 1) entry.key,
    };

    final cellBlocks = <String>[];

    for (var cellIdx = 0; cellIdx < selectedCells.length; cellIdx++) {
      final cell = selectedCells[cellIdx];
      final rawTrimmed = cell.code.trim();
      if (rawTrimmed.isEmpty) continue;

      final runIdx = includedCellIds.length;
      includedCellIds.add(cell.id);

      // Check for %pub add or pub add magic commands.
      final pubAddMatch = RegExp(
        r'^(?:%)?(?:pub\s+add|add)\s+([\w\d_\-]+)\s*;?$',
      ).firstMatch(rawTrimmed);
      if (pubAddMatch != null) {
        final pkgName = pubAddMatch.group(1)!;
        final escapedPkg = _escapeDartString(pkgName);
        cellBlocks.add('''
      // --- Cell $runIdx ---
      _currentCellIdx = $runIdx;
      clearCapturedOutput();
      _cellValue = 'Serverless Wasm mode includes pre-bundled packages (ndarray, pocketfft, openblas, gpuarray, resource_scope, notebook). Dynamic pub add ("$escapedPkg") requires a local server.';
      _recordCellSuccess(cellIds[$runIdx], _cellValue, cellResults);''');
        continue;
      }

      final extracted = _extractImportsAndBody(cell.code);
      final newlyImportedPkgs = <String>[];
      for (final imp in extracted.imports) {
        var normalized = imp.trim();
        if (normalized.contains('package:notebook/src/kernel_helper.dart')) {
          normalized = "import 'package:notebook/src/notebook_widgets.dart';";
        } else if (normalized == "import 'package:gpuarray/gpuarray.dart';" ||
            normalized == 'import "package:gpuarray/gpuarray.dart";') {
          normalized = _gpuCoreImport;
        }
        if (seenImports.add(normalized)) {
          userImports.add(normalized);
        }
        final pkgMatch = RegExp(
          r'''^import\s+['"]package:([\w\d_\-]+)/''',
        ).firstMatch(normalized);
        if (pkgMatch != null) {
          newlyImportedPkgs.add(pkgMatch.group(1)!);
        } else {
          final dartMatch = RegExp(
            r'''^import\s+['"]dart:([\w\d_\-]+)''',
          ).firstMatch(normalized);
          newlyImportedPkgs.add(dartMatch?.group(1) ?? 'library');
        }
      }

      final items = _splitTopLevelItems(extracted.body);
      if (items.isEmpty) {
        if (newlyImportedPkgs.isNotEmpty) {
          final msg = _escapeDartString(
            'Imported ${newlyImportedPkgs.join(', ')}',
          );
          cellBlocks.add('''
      // --- Cell $runIdx ---
      _currentCellIdx = $runIdx;
      clearCapturedOutput();
      _cellValue = '$msg';
      _recordCellSuccess(cellIds[$runIdx], _cellValue, cellResults);''');
        } else {
          cellBlocks.add('''
      // --- Cell $runIdx ---
      _currentCellIdx = $runIdx;
      clearCapturedOutput();
      _cellValue = null;
      _recordCellSuccess(cellIds[$runIdx], _cellValue, cellResults);''');
        }
        continue;
      }

      final bodyLines = <String>[
        '// --- Cell $runIdx ---',
        '_currentCellIdx = $runIdx;',
        'clearCapturedOutput();',
        '_cellValue = null;',
      ];
      String? singleDeclaredVar;
      String? lastDeclaredSymbol;
      var onlyDeclarations = true;

      for (var i = 0; i < items.length; i++) {
        final item = items[i];
        final isLast = i == items.length - 1;
        final kind = _classifyItem(item);

        switch (kind.type) {
          case _ItemType.typeOrFunctionDeclaration:
            final symName = kind.symbolName ?? 'decl_${cellIdx}_$i';
            var declSource = item.trim();
            if (!declSource.endsWith('}') && !declSource.endsWith(';')) {
              declSource = '$declSource;';
            }
            topLevelDeclarations[symName] = declSource;
            lastDeclaredSymbol = kind.symbolName;
          case _ItemType.variableDeclaration:
            onlyDeclarations = false;
            final anyRedeclarations = kind.variables.any(
              (v) => redeclaredVars.contains(v.name),
            );
            if (!anyRedeclarations) {
              var declStmt = _stripTrailingSemicolon(item.trim());
              if (usesGpuArray) {
                declStmt = _rewriteGpuReadbacks(declStmt);
              }
              bodyLines.add('$declStmt;');
            } else {
              for (final v in kind.variables) {
                if (v.initializer != null) {
                  final initExpr = usesGpuArray
                      ? _rewriteGpuReadbacks(v.initializer!)
                      : v.initializer!;
                  bodyLines.add('${v.name} = $initExpr;');
                }
                bodyLines.add('_has_${v.name} = true;');
              }
            }
            if (usesGpuArray) {
              bodyLines.add('await GpuDevice.synchronizeDefault();');
            }
            for (final v in kind.variables) {
              if (seenVariables.add(v.name)) {
                declaredVariables.add(v.name);
              }
              final escapedName = _escapeDartString(v.name);
              bodyLines.add(
                "variableSnapshots['$escapedName'] = describeVariableForInspector('$escapedName', ${v.name});",
              );
            }
            if (items.length == 1 && kind.variables.length == 1) {
              singleDeclaredVar = kind.variables.first.name;
            }
          case _ItemType.statement:
            onlyDeclarations = false;
            var stmtSource = item.trim();
            if (usesGpuArray) {
              stmtSource = _rewriteGpuReadbacks(stmtSource);
            }
            if (!stmtSource.endsWith('}') && !stmtSource.endsWith(';')) {
              stmtSource = '$stmtSource;';
            }
            bodyLines.add(stmtSource);
            if (usesGpuArray) {
              bodyLines.add('await GpuDevice.synchronizeDefault();');
            }
          case _ItemType.expression:
            onlyDeclarations = false;
            var exprSource = _stripTrailingSemicolon(item.trim());
            if (usesGpuArray) {
              exprSource = _rewriteGpuReadbacks(exprSource);
            }
            if (isLast) {
              bodyLines.add(
                '_cellValue = await (() async => (\n$exprSource\n))();',
              );
            } else {
              bodyLines.add('$exprSource;');
            }
            if (usesGpuArray) {
              bodyLines.add('await GpuDevice.synchronizeDefault();');
            }
        }
      }

      if (singleDeclaredVar != null) {
        final escapedName = _escapeDartString(singleDeclaredVar);
        bodyLines.add('''
{
  final _valStr = prettyFormat($singleDeclaredVar);
  final _printed = getCapturedOutput().trim();
  clearCapturedOutput();
  final _header = 'Declared variable $escapedName\\nValue: \$_valStr';
  _cellValue = _printed.isNotEmpty ? '\$_printed\\n\$_header' : _header;
}''');
      } else if (onlyDeclarations && lastDeclaredSymbol != null) {
        final escapedSym = _escapeDartString(lastDeclaredSymbol);
        bodyLines.add("_cellValue = 'Declared: $escapedSym';");
      }

      bodyLines.add(
        '_recordCellSuccess(cellIds[$runIdx], _cellValue, cellResults);',
      );
      cellBlocks.add(bodyLines.map((l) => '      $l').join('\n'));
    }

    final sb = StringBuffer();
    sb.writeln(
      '// ignore_for_file: unused_import, unused_local_variable, unused_element, prefer_typing_uninitialized_variables',
    );
    for (final imp in _defaultImports) {
      sb.writeln(imp);
    }
    for (final imp in userImports) {
      sb.writeln(imp);
    }
    sb.writeln();
    sb.writeln("@JS('notebookReportRunResult')");
    sb.writeln('external void _notebookReportRunResult(JSString jsonPayload);');
    sb.writeln();

    for (final decl in topLevelDeclarations.values) {
      sb.writeln(decl);
      sb.writeln();
    }

    sb.writeln('''
void _recordCellSuccess(
  String cellId,
  dynamic value,
  List<Map<String, Object?>> cellResults,
) {
  final outputs = List<CellOutputItem>.of(capturedOutputs);
  final evalItem = formatEvaluationValue(value);
  if (evalItem != null) {
    outputs.add(evalItem);
  }
  cellResults.add({
    'cellId': cellId,
    'isError': false,
    'output': outputs.map((e) => e.data).join('\\n'),
    'outputs': outputs.map((e) => e.toJson()).toList(),
  });
}
''');

    sb.writeln('Future<void> main() async {');
    sb.writeln('  final cellResults = <Map<String, Object?>>[];');
    sb.writeln('  final variableSnapshots = <String, Map<String, String>>{};');
    sb.writeln('  final cellIds = <String>[');
    for (final id in includedCellIds) {
      sb.writeln("    '${_escapeDartString(id)}',");
    }
    sb.writeln('  ];');
    for (final varName in redeclaredVars) {
      sb.writeln('  dynamic $varName;');
      sb.writeln('  var _has_$varName = false;');
    }
    sb.writeln('  var _currentCellIdx = 0;');
    sb.writeln('  dynamic _cellValue;');
    sb.writeln('  await evalInNotebookZone(() async {');
    sb.writeln('    try {');
    if (usesGpuArray) {
      sb.writeln('      await GpuDevice.ensureDefaultInitialized();');
    }
    for (final block in cellBlocks) {
      sb.writeln(block);
    }
    for (final varName in declaredVariables) {
      final escaped = _escapeDartString(varName);
      if (redeclaredVars.contains(varName)) {
        sb.writeln(
          "      if (_has_$varName) variableSnapshots['$escaped'] = describeVariableForInspector('$escaped', $varName);",
        );
      } else {
        sb.writeln(
          "      variableSnapshots['$escaped'] = describeVariableForInspector('$escaped', $varName);",
        );
      }
    }
    sb.writeln('''
    } catch (e, st) {
      if (_currentCellIdx >= 0 && _currentCellIdx < cellIds.length) {
        cellResults.add({
          'cellId': cellIds[_currentCellIdx],
          'isError': true,
          'output': '\$e\\n\$st'.trim(),
          'outputs': const <Map<String, String>>[],
        });
      }
    }
  });
  final payload = jsonEncode({
    'cells': cellResults,
    'variables': variableSnapshots.values.toList(),
  });
  _notebookReportRunResult(payload.toJS);
}
''');

    return WasmBundledProgram(
      mainDartSource: sb.toString(),
      includedCellIds: includedCellIds,
      declaredVariables: declaredVariables,
    );
  }

  /// Bundles [cells] into a single Dart source file for LSP completion/hover
  /// and computes the mapped offset of [cursorOffsetInCell] within [activeCellId].
  static WasmAnalysisBundle bundleForAnalysis(
    List<WasmNotebookCell> cells, {
    required String activeCellId,
    required int cursorOffsetInCell,
  }) {
    final sb = StringBuffer();
    sb.writeln(
      '// ignore_for_file: unused_import, unused_local_variable, unused_element',
    );
    for (final imp in _defaultImports) {
      sb.writeln(imp);
    }

    final seenImports = <String>{..._defaultImports};
    if (cells.any((c) => c.type == 'code' && _gpuUsageRegex.hasMatch(c.code))) {
      if (seenImports.add(_gpuCoreImport)) {
        sb.writeln(_gpuCoreImport);
      }
    }
    final priorTopDecls = <String>[];
    final priorVarDecls = <String>[];
    var activeCellCode = '';

    for (final cell in cells) {
      if (cell.type != 'code') continue;
      if (cell.id == activeCellId) {
        activeCellCode = cell.code;
        break;
      }
      final extracted = _extractImportsAndBody(cell.code);
      for (final imp in extracted.imports) {
        final norm = imp.trim();
        if (seenImports.add(norm)) {
          sb.writeln(norm);
        }
      }
      for (final item in _splitTopLevelItems(extracted.body)) {
        final kind = _classifyItem(item);
        if (kind.type == _ItemType.typeOrFunctionDeclaration) {
          var d = item.trim();
          if (!d.endsWith('}') && !d.endsWith(';')) d = '$d;';
          priorTopDecls.add(d);
        } else if (kind.type == _ItemType.variableDeclaration) {
          var d = item.trim();
          if (!d.endsWith(';')) d = '$d;';
          priorVarDecls.add(d);
        }
      }
    }

    sb.writeln();
    for (final d in priorTopDecls) {
      sb.writeln(d);
    }
    sb.writeln('Future<void> __notebookAnalysisScope() async {');
    for (final v in priorVarDecls) {
      sb.writeln('  $v');
    }
    final prefixLength = sb.length;
    final clampedCursor = cursorOffsetInCell.clamp(0, activeCellCode.length);
    sb.writeln(activeCellCode);
    sb.writeln('}');

    return WasmAnalysisBundle(
      source: sb.toString(),
      mappedOffset: prefixLength + clampedCursor,
    );
  }

  static ({List<String> imports, String body}) _extractImportsAndBody(
    String code,
  ) {
    final imports = <String>[];
    final importRegex = RegExp(
      r'''^\s*import\s+['"][^;]+;\s*''',
      multiLine: true,
    );
    for (final match in importRegex.allMatches(code)) {
      imports.add(match.group(0)!.trim());
    }
    final body = code.replaceAll(importRegex, '');
    return (imports: imports, body: body);
  }

  /// Splits [source] into top-level statements and declarations while respecting
  /// comments, string literals (including triple-quoted and raw strings), and
  /// nested `()`, `[]`, `{}` blocks.
  static List<String> _splitTopLevelItems(String source) {
    final items = <String>[];
    final len = source.length;
    var itemStart = 0;
    var parenDepth = 0;
    var bracketDepth = 0;
    var braceDepth = 0;
    var i = 0;

    void flushItem(int endExclusive) {
      final slice = source.substring(itemStart, endExclusive);
      if (_stripLeadingTrivia(slice).isNotEmpty) {
        items.add(slice.trim());
      }
      itemStart = endExclusive;
    }

    while (i < len) {
      final ch = source.codeUnitAt(i);

      // Line comment //
      if (ch == 0x2F /* / */ &&
          i + 1 < len &&
          source.codeUnitAt(i + 1) == 0x2F) {
        i += 2;
        while (i < len && source.codeUnitAt(i) != 0x0A) {
          i++;
        }
        continue;
      }

      // Block comment /* ... */
      if (ch == 0x2F /* / */ &&
          i + 1 < len &&
          source.codeUnitAt(i + 1) == 0x2A) {
        i += 2;
        while (i + 1 < len &&
            !(source.codeUnitAt(i) == 0x2A &&
                source.codeUnitAt(i + 1) == 0x2F)) {
          i++;
        }
        i += 2;
        continue;
      }

      // Raw or normal string literal
      if (ch == 0x27 /* ' */ ||
          ch == 0x22 /* " */ ||
          (ch == 0x72 /* r */ &&
              i + 1 < len &&
              (source.codeUnitAt(i + 1) == 0x27 ||
                  source.codeUnitAt(i + 1) == 0x22))) {
        final isRaw = ch == 0x72;
        if (isRaw) i++;
        final quote = source.codeUnitAt(i);
        final isTriple =
            i + 2 < len &&
            source.codeUnitAt(i + 1) == quote &&
            source.codeUnitAt(i + 2) == quote;
        i += isTriple ? 3 : 1;
        while (i < len) {
          final c = source.codeUnitAt(i);
          if (!isRaw && c == 0x5C /* \ */ ) {
            i += 2;
            continue;
          }
          if (isTriple) {
            if (i + 2 < len &&
                source.codeUnitAt(i) == quote &&
                source.codeUnitAt(i + 1) == quote &&
                source.codeUnitAt(i + 2) == quote) {
              i += 3;
              break;
            }
          } else if (c == quote) {
            i++;
            break;
          }
          i++;
        }
        continue;
      }

      if (ch == 0x28 /* ( */ ) {
        parenDepth++;
      } else if (ch == 0x29 /* ) */ ) {
        if (parenDepth > 0) parenDepth--;
      } else if (ch == 0x5B /* [ */ ) {
        bracketDepth++;
      } else if (ch == 0x5D /* ] */ ) {
        if (bracketDepth > 0) bracketDepth--;
      } else if (ch == 0x7B /* { */ ) {
        braceDepth++;
      } else if (ch == 0x7D /* } */ ) {
        if (braceDepth > 0) braceDepth--;
        if (braceDepth == 0 && parenDepth == 0 && bracketDepth == 0) {
          final currentText = _stripLeadingTrivia(
            source.substring(itemStart, i + 1),
          );
          final nextWord = _peekNextWord(source, i + 1);
          final continuesBlock =
              nextWord == 'else' ||
              nextWord == 'catch' ||
              nextWord == 'on' ||
              nextWord == 'finally' ||
              (nextWord == 'while' && RegExp(r'^do\b').hasMatch(currentText));
          if (!continuesBlock && _isBlockStatementOrDeclaration(currentText)) {
            flushItem(i + 1);
          }
        }
      } else if (ch == 0x3B /* ; */ &&
          braceDepth == 0 &&
          parenDepth == 0 &&
          bracketDepth == 0) {
        flushItem(i + 1);
      }

      i++;
    }

    if (itemStart < len) {
      flushItem(len);
    }

    return items;
  }

  static bool _isBlockStatementOrDeclaration(String strippedText) {
    if (RegExp(
      r'^(?:abstract\s+|base\s+|final\s+|interface\s+|sealed\s+|mixin\s+)*(?:class|enum|mixin|extension)\b',
    ).hasMatch(strippedText)) {
      return true;
    }
    if (RegExp(
      r'^(?:if|for|while|switch|try|await\s+for)\b',
    ).hasMatch(strippedText)) {
      return true;
    }
    return _matchFunctionDeclarationName(strippedText) != null;
  }

  static _ClassifiedItem _classifyItem(String rawItem) {
    final stripped = _stripLeadingTrivia(rawItem).trim();

    // 1. class / enum / mixin / extension / typedef
    final typeMatch = RegExp(
      r'^(?:abstract\s+|base\s+|final\s+|interface\s+|sealed\s+|mixin\s+)*(?:class|enum|mixin|extension|typedef)\s+([A-Za-z_$][\w$]*)',
    ).firstMatch(stripped);
    if (typeMatch != null) {
      return _ClassifiedItem(
        type: _ItemType.typeOrFunctionDeclaration,
        symbolName: typeMatch.group(1),
      );
    }
    if (RegExp(r'^extension\s+on\b').hasMatch(stripped)) {
      return const _ClassifiedItem(type: _ItemType.typeOrFunctionDeclaration);
    }

    // 2. Top-level function declaration: [returnType] name([params]) [async] { or =>
    final fnName = _matchFunctionDeclarationName(stripped);
    if (fnName != null) {
      return _ClassifiedItem(
        type: _ItemType.typeOrFunctionDeclaration,
        symbolName: fnName,
      );
    }

    // 3. Control-flow / jump / block statement
    final firstWordMatch = RegExp(
      r'^([A-Za-z_$][\w$]*)\b',
    ).firstMatch(stripped);
    final firstWord = firstWordMatch?.group(1);
    if (firstWord != null &&
        _statementKeywords.contains(firstWord) &&
        firstWord != 'await') {
      return const _ClassifiedItem(type: _ItemType.statement);
    }

    // 4. Variable declaration: (late)? (var|final|const|<Type>) name = ...
    final varDecls = _tryParseVariableDeclaration(stripped);
    if (varDecls != null && varDecls.isNotEmpty) {
      return _ClassifiedItem(
        type: _ItemType.variableDeclaration,
        variables: varDecls,
      );
    }

    // 5. Otherwise it is an expression statement.
    return const _ClassifiedItem(type: _ItemType.expression);
  }

  static String? _matchFunctionDeclarationName(String stripped) {
    final headerMatch = RegExp(
      r'^(?:[A-Za-z_$][\w$]*(?:\s*<[^>]+>)?\??\s+)?([A-Za-z_$][\w$]*)\s*(?:<[^>]+>)?\s*\(',
    ).firstMatch(stripped);
    if (headerMatch == null) return null;
    final name = headerMatch.group(1)!;
    if (_statementKeywords.contains(name) ||
        name == 'var' ||
        name == 'final' ||
        name == 'const' ||
        name == 'print' ||
        name == 'display') {
      return null;
    }
    // Find matching ')' for the parameter list '('
    final openParenIdx = stripped.indexOf('(', headerMatch.start);
    if (openParenIdx == -1) return null;
    var depth = 0;
    var closeParenIdx = -1;
    for (var i = openParenIdx; i < stripped.length; i++) {
      final c = stripped.codeUnitAt(i);
      if (c == 0x28 /* ( */ ) {
        depth++;
      } else if (c == 0x29 /* ) */ ) {
        depth--;
        if (depth == 0) {
          closeParenIdx = i;
          break;
        }
      }
    }
    if (closeParenIdx == -1) return null;
    final afterParams = stripped.substring(closeParenIdx + 1).trimLeft();
    if (RegExp(
      r'^(?:(?:async|sync)\s*\*?\s*)?(?:\{|=>)',
    ).hasMatch(afterParams)) {
      return name;
    }
    return null;
  }

  static List<_ParsedVarDecl>? _tryParseVariableDeclaration(String stripped) {
    final withoutSemi = _stripTrailingSemicolon(stripped).trim();
    String? declaratorsPart;

    final kwMatch = RegExp(
      r'^(?:late\s+)?(?:var|(?:final|const)(?:\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?(?:\s*<[^>]+>)?\??(?=\s+[A-Za-z_$]))?)\s+(.+)$',
      dotAll: true,
    ).firstMatch(withoutSemi);
    if (kwMatch != null) {
      declaratorsPart = kwMatch.group(1)!;
    } else {
      final typedMatch = RegExp(
        r'^(?:late\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?(?:\s*<[^>]+>)?\??)\s+([A-Za-z_$][\w$]*\s*(?:=.*)?)$',
        dotAll: true,
      ).firstMatch(withoutSemi);
      if (typedMatch != null) {
        final typeToken = typedMatch.group(1)!.split(RegExp(r'[\s<]')).first;
        if (!_statementKeywords.contains(typeToken)) {
          declaratorsPart = typedMatch.group(2)!;
        }
      }
    }

    if (declaratorsPart == null) return null;
    // Reject record destructuring patterns like `final (a, b) = ...` here so they
    // stay as statements or handle simple identifier declarators.
    if (declaratorsPart.trimLeft().startsWith('(')) return null;

    final parts = _splitTopLevelComma(declaratorsPart);
    final result = <_ParsedVarDecl>[];
    for (final part in parts) {
      final trimmed = part.trim();
      final eqIdx = _findTopLevelEquals(trimmed);
      if (eqIdx == -1) {
        if (RegExp(r'^[A-Za-z_$][\w$]*$').hasMatch(trimmed)) {
          result.add(_ParsedVarDecl(name: trimmed));
        } else {
          return null;
        }
      } else {
        final lhs = trimmed.substring(0, eqIdx).trim();
        final rhs = trimmed.substring(eqIdx + 1).trim();
        if (!RegExp(r'^[A-Za-z_$][\w$]*$').hasMatch(lhs) || rhs.isEmpty) {
          return null;
        }
        result.add(_ParsedVarDecl(name: lhs, initializer: rhs));
      }
    }
    return result;
  }

  static List<String> _splitTopLevelComma(String s) {
    final parts = <String>[];
    var start = 0;
    var depth = 0;
    for (var i = 0; i < s.length; i++) {
      final c = s.codeUnitAt(i);
      if (c == 0x28 || c == 0x5B || c == 0x7B) {
        depth++;
      } else if (c == 0x29 || c == 0x5D || c == 0x7D) {
        if (depth > 0) depth--;
      } else if (c == 0x2C /* , */ && depth == 0) {
        parts.add(s.substring(start, i));
        start = i + 1;
      }
    }
    parts.add(s.substring(start));
    return parts;
  }

  static int _findTopLevelEquals(String s) {
    var depth = 0;
    for (var i = 0; i < s.length; i++) {
      final c = s.codeUnitAt(i);
      if (c == 0x28 || c == 0x5B || c == 0x7B) {
        depth++;
      } else if (c == 0x29 || c == 0x5D || c == 0x7D) {
        if (depth > 0) depth--;
      } else if (c == 0x3D /* = */ && depth == 0) {
        final prev = i > 0 ? s.codeUnitAt(i - 1) : 0;
        final next = i + 1 < s.length ? s.codeUnitAt(i + 1) : 0;
        if (next != 0x3D /* == */ &&
            next != 0x3E /* => */ &&
            prev != 0x21 /* != */ &&
            prev != 0x3C /* <= */ &&
            prev != 0x3E /* >= */ ) {
          return i;
        }
      }
    }
    return -1;
  }

  static String _stripLeadingTrivia(String s) {
    var i = 0;
    final len = s.length;
    while (i < len) {
      final c = s.codeUnitAt(i);
      if (c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D) {
        i++;
        continue;
      }
      if (c == 0x2F && i + 1 < len && s.codeUnitAt(i + 1) == 0x2F) {
        i += 2;
        while (i < len && s.codeUnitAt(i) != 0x0A) {
          i++;
        }
        continue;
      }
      if (c == 0x2F && i + 1 < len && s.codeUnitAt(i + 1) == 0x2A) {
        i += 2;
        while (i + 1 < len &&
            !(s.codeUnitAt(i) == 0x2A && s.codeUnitAt(i + 1) == 0x2F)) {
          i++;
        }
        i += 2;
        continue;
      }
      break;
    }
    return s.substring(i);
  }

  static String? _peekNextWord(String s, int fromIndex) {
    final rest = _stripLeadingTrivia(s.substring(fromIndex));
    final m = RegExp(r'^([A-Za-z_$][\w$]*)\b').firstMatch(rest);
    return m?.group(1);
  }

  static String _stripTrailingSemicolon(String s) {
    final trimmed = s.trimRight();
    if (trimmed.endsWith(';')) {
      return trimmed.substring(0, trimmed.length - 1).trimRight();
    }
    return trimmed;
  }

  static const String _gpuCoreImport =
      "import 'package:gpuarray/gpuarray.dart' show "
      'BrowserWebGpuBackend, GradFn, GpuArray, GpuArrayNDArrayInterop, '
      'GpuBackend, GpuBuffer, GpuBufferUsage, GpuDevice, '
      'GpuDeviceDisposedException, GpuDeviceException, GpuDeviceType, '
      'GpuException, GpuMemoryException, GpuMemoryPool, '
      'GpuShaderCompilationException, GpuShapeMismatchException, GpuSlice, '
      'LossReduction, NDArrayGpuInterop, createDefaultGpuBackend, '
      'createWebGpuDevice, enableGrad, isGradEnabled, noGrad;';

  static final RegExp _gpuUsageRegex = RegExp(
    r'\b(?:Gpu\w*|WebGpu\w*|wgsl\w*|createWebGpuDevice)\b|\.\s*toGpu\s*\(',
  );

  static final RegExp _gpuReadbackCallRegex = RegExp(
    r'\.\s*to(?:Host)?NDArray\s*\(\s*\)',
  );

  static String _rewriteGpuReadbacks(String code) {
    var current = code;
    while (true) {
      final match = _gpuReadbackCallRegex.firstMatch(current);
      if (match == null) break;
      final dotIndex = match.start;
      final recvStart = _findReceiverStart(current, dotIndex);
      if (recvStart >= dotIndex) break;
      final receiver = current.substring(recvStart, dotIndex);
      current =
          '${current.substring(0, recvStart)}(await ($receiver).toNDArrayAsync())${current.substring(match.end)}';
    }
    return current;
  }

  static bool _isIdentChar(int c) =>
      (c >= 0x30 && c <= 0x39) ||
      (c >= 0x41 && c <= 0x5A) ||
      (c >= 0x61 && c <= 0x7A) ||
      c == 0x5F ||
      c == 0x24;

  static int _findReceiverStart(String s, int dotIndex) {
    var j = dotIndex - 1;
    var start = dotIndex;
    while (j >= 0) {
      while (j >= 0 &&
          (s.codeUnitAt(j) == 0x20 ||
              s.codeUnitAt(j) == 0x09 ||
              s.codeUnitAt(j) == 0x0A ||
              s.codeUnitAt(j) == 0x0D)) {
        j--;
      }
      if (j < 0) break;

      final c = s.codeUnitAt(j);
      if (c == 0x29 /* ) */ || c == 0x5D /* ] */ ) {
        final openChar = c == 0x29 ? 0x28 : 0x5B;
        var depth = 1;
        j--;
        while (j >= 0 && depth > 0) {
          final ch = s.codeUnitAt(j);
          if (ch == c) {
            depth++;
          } else if (ch == openChar) {
            depth--;
          }
          j--;
        }
        start = j + 1;
        var k = j;
        while (k >= 0 && (s.codeUnitAt(k) == 0x20 || s.codeUnitAt(k) == 0x09)) {
          k--;
        }
        if (k >= 0 && s.codeUnitAt(k) == 0x3E /* > */ ) {
          var angleDepth = 1;
          k--;
          while (k >= 0 && angleDepth > 0) {
            final ch = s.codeUnitAt(k);
            if (ch == 0x3E) {
              angleDepth++;
            } else if (ch == 0x3C /* < */ ) {
              angleDepth--;
            }
            k--;
          }
          if (angleDepth == 0) {
            j = k;
            start = j + 1;
          }
        }
        continue;
      }

      if (_isIdentChar(c)) {
        while (j >= 0 && _isIdentChar(s.codeUnitAt(j))) {
          j--;
        }
        final word = s.substring(j + 1, start);
        if (_statementKeywords.contains(word) ||
            word == 'final' ||
            word == 'var' ||
            word == 'const') {
          break;
        }
        start = j + 1;
        var k = j;
        while (k >= 0 &&
            (s.codeUnitAt(k) == 0x20 ||
                s.codeUnitAt(k) == 0x09 ||
                s.codeUnitAt(k) == 0x0A ||
                s.codeUnitAt(k) == 0x0D)) {
          k--;
        }
        if (k >= 0 && s.codeUnitAt(k) == 0x2E /* . */ ) {
          j = k - 1;
          start = k;
          continue;
        }
        break;
      }

      break;
    }
    return start;
  }

  static String _escapeDartString(String s) => s
      .replaceAll(r'\', r'\\')
      .replaceAll("'", r"\'")
      .replaceAll(r'$', r'\$')
      .replaceAll('\n', r'\n')
      .replaceAll('\r', r'\r');
}

enum _ItemType {
  typeOrFunctionDeclaration,
  variableDeclaration,
  statement,
  expression,
}

final class _ParsedVarDecl {
  final String name;
  final String? initializer;
  const _ParsedVarDecl({required this.name, this.initializer});
}

final class _ClassifiedItem {
  final _ItemType type;
  final String? symbolName;
  final List<_ParsedVarDecl> variables;
  const _ClassifiedItem({
    required this.type,
    this.symbolName,
    this.variables = const [],
  });
}
