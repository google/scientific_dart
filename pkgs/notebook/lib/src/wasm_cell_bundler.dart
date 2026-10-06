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

/// Classification of a top-level item inside a parsed notebook cell.
enum ParsedCellItemType {
  /// A `class`, `enum`, `mixin`, `extension`, `typedef`, or function declaration.
  typeOrFunctionDeclaration,

  /// A cell-level variable declaration (`var`, `final`, `const`, or typed).
  variableDeclaration,

  /// A control-flow, jump, or pattern-declaration statement.
  statement,

  /// An expression statement (whose value is captured when it is the trailing item).
  expression,
}

/// An `import` directive extracted from a notebook cell.
final class ParsedCellImport {
  /// The trimmed `import '...';` directive source.
  final String statement;

  /// Short package or library name (for example `'path'` for `package:path/...`
  /// or `'math'` for `dart:math`).
  final String name;

  /// Constructs a [ParsedCellImport].
  const ParsedCellImport({required this.statement, required this.name});
}

/// A single variable declarator (`name` and optional `initializer`) within a
/// cell variable declaration.
final class ParsedVarDeclarator {
  /// The declared variable identifier.
  final String name;

  /// The initializer expression source, or `null` if uninitialized.
  final String? initializer;

  /// Constructs a [ParsedVarDeclarator].
  const ParsedVarDeclarator({required this.name, this.initializer});
}

/// A single classified top-level item from a notebook cell body.
final class ParsedCellItem {
  /// The classification of this item.
  final ParsedCellItemType type;

  /// Normalized source text of this item.
  ///
  /// For [ParsedCellItemType.typeOrFunctionDeclaration] and
  /// [ParsedCellItemType.statement], ends with `;` or `}`.
  /// For [ParsedCellItemType.variableDeclaration] and
  /// [ParsedCellItemType.expression], any trailing `;` is stripped.
  final String source;

  /// Declared symbol name for [ParsedCellItemType.typeOrFunctionDeclaration], if named.
  final String? symbolName;

  /// Whether a [ParsedCellItemType.variableDeclaration] uses `const`.
  final bool isConst;

  /// Whether a [ParsedCellItemType.variableDeclaration] uses `final`.
  final bool isFinal;

  /// Explicit type annotation on a [ParsedCellItemType.variableDeclaration], if any.
  final String? typeAnnotation;

  /// Declarators for [ParsedCellItemType.variableDeclaration].
  final List<ParsedVarDeclarator> variables;

  /// Constructs a [ParsedCellItem].
  const ParsedCellItem({
    required this.type,
    required this.source,
    this.symbolName,
    this.isConst = false,
    this.isFinal = false,
    this.typeAnnotation,
    this.variables = const [],
  });
}

/// Result of transforming notebook cell code into top-level workspace
/// definitions and an executable cell body.
final class CellTransformationResult {
  /// Top-level definitions to register in the workspace.
  final List<String> topLevelDefinitions;

  /// Top-level definitions keyed by symbol name.
  final Map<String, String> namedDefinitions;

  /// Variable names declared in this cell.
  final List<String> declaredVariables;

  /// The transformed body code for `async` cell execution.
  final String cellBodyCode;

  /// Constructs a [CellTransformationResult].
  const CellTransformationResult(
    this.topLevelDefinitions,
    this.cellBodyCode, {
    this.namedDefinitions = const {},
    this.declaredVariables = const [],
  });
}

/// Parsed and classified representation of a single notebook code cell,
/// shared between [WasmCellBundler] (serverless Wasm mode) and
/// `NotebookKernel` (VM-service kernel mode).
final class ParsedNotebookCell {
  /// Package name if this cell is a `%pub add <pkg>` / `pub add <pkg>` magic command.
  final String? pubAddPackage;

  /// `import` directives extracted from the cell.
  final List<ParsedCellImport> imports;

  /// Ordered top-level statements, declarations, and expressions in the cell body.
  final List<ParsedCellItem> items;

  /// Constructs a [ParsedNotebookCell].
  const ParsedNotebookCell({
    this.pubAddPackage,
    this.imports = const [],
    this.items = const [],
  });

  /// Parses [code] into a [ParsedNotebookCell].
  factory ParsedNotebookCell.parse(String code) =>
      WasmCellBundler._parseCell(code);

  /// Whether the cell contains no magic command, imports, or body items.
  bool get isEmpty => pubAddPackage == null && imports.isEmpty && items.isEmpty;

  /// Whether the cell consists solely of `import` directives.
  bool get isPureImport =>
      pubAddPackage == null && imports.isNotEmpty && items.isEmpty;

  /// Human-readable status string for a pure-import cell (`'Imported ...'`).
  String get importedNamesSummary =>
      'Imported ${imports.map((i) => i.name).join(', ')}';

  /// Whether the cell body contains only type or function declarations.
  bool get onlyDeclarations =>
      items.isNotEmpty &&
      items.every(
        (i) => i.type == ParsedCellItemType.typeOrFunctionDeclaration,
      );

  /// The last named symbol among [ParsedCellItemType.typeOrFunctionDeclaration] items.
  String? get lastDeclaredSymbol {
    for (var i = items.length - 1; i >= 0; i--) {
      final item = items[i];
      if (item.type == ParsedCellItemType.typeOrFunctionDeclaration &&
          item.symbolName != null) {
        return item.symbolName;
      }
    }
    return null;
  }

  /// If the cell body consists of a single variable declaration of a single
  /// variable, returns that [ParsedVarDeclarator]; otherwise `null`.
  ParsedVarDeclarator? get singleDeclaredVariable {
    if (items.length == 1) {
      final item = items.first;
      if (item.type == ParsedCellItemType.variableDeclaration &&
          item.variables.length == 1) {
        return item.variables.first;
      }
    }
    return null;
  }

  /// Transforms this cell's items into top-level workspace definitions and an
  /// `async` function body for execution in `NotebookKernel`.
  CellTransformationResult transformForKernel() =>
      WasmCellBundler._transformForKernel(this);
}

/// Bundles multi-cell notebook sessions into a single compilable Dart program
/// for serverless Wasm execution (re-run-from-top state model) and LSP analysis,
/// and provides the shared cell parser/transformer used by `NotebookKernel`.
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
    if (selectedCells.any(
      (c) => RegExp(r'\b(?:Gpu\w*|WebGpu\w*|wgsl\w*)\b').hasMatch(c.code),
    )) {
      const gpuImport = "import 'package:gpuarray/gpuarray.dart';";
      seenImports.add(gpuImport);
      userImports.add(gpuImport);
    }
    final topLevelDeclarations = <String, String>{};
    final declaredVariables = <String>[];
    final seenVariables = <String>{};
    final varDeclCounts = <String, int>{};
    final includedCellIds = <String>[];

    final parsedCells = <({WasmNotebookCell cell, ParsedNotebookCell parsed})>[
      for (final cell in selectedCells)
        if (cell.code.trim().isNotEmpty)
          (cell: cell, parsed: _parseCell(cell.code)),
    ];

    // Pre-pass: count variable declarations across cells so uniquely declared
    // variables keep their exact Dart static type inference (e.g. NDArray<Float64>)
    // in the shared main() scope, while variables re-declared across multiple
    // cells fall back to a shared `dynamic` slot.
    for (final (:parsed, cell: _) in parsedCells) {
      for (final item in parsed.items) {
        if (item.type == ParsedCellItemType.variableDeclaration) {
          for (final v in item.variables) {
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

    for (var cellIdx = 0; cellIdx < parsedCells.length; cellIdx++) {
      final (:cell, :parsed) = parsedCells[cellIdx];
      final runIdx = includedCellIds.length;
      includedCellIds.add(cell.id);

      if (parsed.pubAddPackage case final pkgName?) {
        final escapedPkg = _escapeDartString(pkgName);
        cellBlocks.add('''
      // --- Cell $runIdx ---
      _currentCellIdx = $runIdx;
      clearCapturedOutput();
      _cellValue = 'Serverless Wasm mode includes pre-bundled packages (ndarray, pocketfft, openblas, gpuarray, resource_scope, notebook). Dynamic pub add ("$escapedPkg") requires a local server.';
      _recordCellSuccess(cellIds[$runIdx], _cellValue, cellResults);''');
        continue;
      }

      for (final imp in parsed.imports) {
        var normalized = imp.statement;
        if (normalized.contains('package:notebook/src/kernel_helper.dart')) {
          normalized = "import 'package:notebook/src/notebook_widgets.dart';";
        }
        if (seenImports.add(normalized)) {
          userImports.add(normalized);
        }
      }

      if (parsed.items.isEmpty) {
        if (parsed.isPureImport) {
          final msg = _escapeDartString(parsed.importedNamesSummary);
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

      for (var i = 0; i < parsed.items.length; i++) {
        final item = parsed.items[i];
        final isLast = i == parsed.items.length - 1;

        switch (item.type) {
          case ParsedCellItemType.typeOrFunctionDeclaration:
            final symName = item.symbolName ?? 'decl_${cellIdx}_$i';
            topLevelDeclarations[symName] = item.source;
          case ParsedCellItemType.variableDeclaration:
            final anyRedeclarations = item.variables.any(
              (v) => redeclaredVars.contains(v.name),
            );
            if (!anyRedeclarations) {
              bodyLines.add('${item.source};');
            } else {
              for (final v in item.variables) {
                if (v.initializer != null) {
                  bodyLines.add('${v.name} = ${v.initializer};');
                }
                bodyLines.add('_has_${v.name} = true;');
              }
            }
            for (final v in item.variables) {
              if (seenVariables.add(v.name)) {
                declaredVariables.add(v.name);
              }
              final escapedName = _escapeDartString(v.name);
              bodyLines.add(
                "variableSnapshots['$escapedName'] = describeVariableForInspector('$escapedName', ${v.name});",
              );
            }
          case ParsedCellItemType.statement:
            bodyLines.add(item.source);
          case ParsedCellItemType.expression:
            if (isLast) {
              // The thunk goes through `_evaluateCellExpression` rather than
              // being awaited directly so that a trailing `void` expression
              // (`print(...)`, `display(...)`, `list.add(...)`, ...) compiles;
              // see the helper's documentation in the generated program.
              bodyLines.add(
                '_cellValue = await _evaluateCellExpression(() async => (\n${item.source}\n));',
              );
            } else {
              bodyLines.add('${item.source};');
            }
        }
      }

      if (parsed.singleDeclaredVariable case final singleVar?) {
        bodyLines.add(
          _buildSingleDeclaredVariableBlock(singleVar, assignToCellValue: true),
        );
      } else if (parsed.onlyDeclarations && parsed.lastDeclaredSymbol != null) {
        final escapedSym = _escapeDartString(parsed.lastDeclaredSymbol!);
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
/// Evaluates the trailing expression of a cell and awaits its value.
///
/// Taking the thunk as a `Future<dynamic> Function()` lets a cell end in a
/// `void` expression such as `print(...)`, `display(...)` or `list.add(...)`:
/// the closure's return type `Future<void>` is a subtype of
/// `Future<dynamic>`, so awaiting the call yields `null` typed as `dynamic`.
/// Awaiting the closure directly would instead yield a `void` value, which is
/// a compile-time error to assign. A trailing `Future` is still awaited.
Future<dynamic> _evaluateCellExpression(Future<dynamic> Function() thunk) =>
    thunk();

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
    final priorTopDecls = <String>[];
    final priorVarDecls = <String>[];
    var activeCellCode = '';

    for (final cell in cells) {
      if (cell.type != 'code') continue;
      if (cell.id == activeCellId) {
        activeCellCode = cell.code;
        break;
      }
      final parsed = _parseCell(cell.code);
      for (final imp in parsed.imports) {
        if (seenImports.add(imp.statement)) {
          sb.writeln(imp.statement);
        }
      }
      for (final item in parsed.items) {
        if (item.type == ParsedCellItemType.typeOrFunctionDeclaration) {
          priorTopDecls.add(item.source);
        } else if (item.type == ParsedCellItemType.variableDeclaration) {
          priorVarDecls.add('${item.source};');
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

  /// Transforms [cell] into workspace top-level definitions and an `async`
  /// cell body for execution in `NotebookKernel`.
  static CellTransformationResult _transformForKernel(ParsedNotebookCell cell) {
    final topLevelDefs = <String>[];
    final namedDefs = <String, String>{};
    final declaredVars = <String>[];
    final bodyLines = <String>[];

    for (var i = 0; i < cell.items.length; i++) {
      final item = cell.items[i];
      final isLast = i == cell.items.length - 1;

      switch (item.type) {
        case ParsedCellItemType.typeOrFunctionDeclaration:
          final symName = item.symbolName ?? 'decl_$i';
          topLevelDefs.add(item.source);
          namedDefs[symName] = item.source;
        case ParsedCellItemType.variableDeclaration:
          final typeAnnotation = item.typeAnnotation;
          for (final v in item.variables) {
            final varName = v.name;
            declaredVars.add(varName);
            final String def;
            if (typeAnnotation != null) {
              def =
                  'dynamic __slot_$varName;\n'
                  '$typeAnnotation get $varName => __slot_$varName as $typeAnnotation;\n'
                  'set $varName($typeAnnotation v) {\n'
                  '  __slot_$varName = v;\n'
                  '}\n'
                  'T __set_$varName<T extends $typeAnnotation>(T v) {\n'
                  '  __slot_$varName = v;\n'
                  '  return v;\n'
                  '}';
            } else {
              def =
                  'dynamic $varName;\n'
                  'T __set_$varName<T>(T v) {\n'
                  '  $varName = v;\n'
                  '  return v;\n'
                  '}';
            }
            topLevelDefs.add(def);
            namedDefs[varName] = def;
            if (v.initializer != null) {
              if (item.isConst) {
                bodyLines.add('const $varName = ${v.initializer};');
                bodyLines.add('__set_$varName($varName);');
              } else {
                final kw = item.isFinal ? 'final' : 'var';
                bodyLines.add(
                  '$kw $varName = __set_$varName(${v.initializer});',
                );
              }
            } else {
              bodyLines.add('// $varName');
            }
          }
        case ParsedCellItemType.statement:
          bodyLines.add(item.source);
        case ParsedCellItemType.expression:
          if (isLast) {
            bodyLines.add(
              'return await evaluateCellExpression(() async => (\n${item.source}\n));',
            );
          } else {
            bodyLines.add('${item.source};');
          }
      }
    }

    if (cell.singleDeclaredVariable case final singleVar?) {
      bodyLines.add(
        _buildSingleDeclaredVariableBlock(singleVar, assignToCellValue: false),
      );
    } else if (cell.onlyDeclarations && cell.lastDeclaredSymbol != null) {
      final escapedSym = _escapeDartString(cell.lastDeclaredSymbol!);
      bodyLines.add("return 'Declared: $escapedSym';");
    }

    return CellTransformationResult(
      topLevelDefs,
      bodyLines.join('\n'),
      namedDefinitions: namedDefs,
      declaredVariables: declaredVars,
    );
  }

  static String _buildSingleDeclaredVariableBlock(
    ParsedVarDeclarator variable, {
    required bool assignToCellValue,
  }) {
    final escapedName = _escapeDartString(variable.name);
    if (variable.initializer == null) {
      return assignToCellValue
          ? "_cellValue = 'Declared variable $escapedName';"
          : "return 'Declared variable $escapedName';";
    }
    final sink = assignToCellValue ? '_cellValue =' : 'return';
    return '''
{
  final _valStr = prettyFormat(${variable.name});
  final _printed = getCapturedOutput().trim();
  clearCapturedOutput();
  final _header = 'Declared variable $escapedName\\nValue: \$_valStr';
  $sink _printed.isNotEmpty ? '\$_printed\\n\$_header' : _header;
}''';
  }

  static ParsedNotebookCell _parseCell(String code) {
    final rawTrimmed = code.trim();
    if (rawTrimmed.isEmpty) {
      return const ParsedNotebookCell();
    }

    final pubAddMatch = RegExp(
      r'^(?:%)?(?:pub\s+add|add)\s+([\w\d_\-]+)\s*;?$',
    ).firstMatch(rawTrimmed);
    if (pubAddMatch != null) {
      return ParsedNotebookCell(pubAddPackage: pubAddMatch.group(1));
    }

    final extracted = _extractImportsAndBody(code);
    final rawItems = _splitTopLevelItems(extracted.body);
    final items = <ParsedCellItem>[
      for (final rawItem in rawItems) _classifyItem(rawItem),
    ];
    return ParsedNotebookCell(imports: extracted.imports, items: items);
  }

  static ({List<ParsedCellImport> imports, String body}) _extractImportsAndBody(
    String code,
  ) {
    final imports = <ParsedCellImport>[];
    final importRegex = RegExp(
      r'''^\s*import\s+['"][^;]+;\s*''',
      multiLine: true,
    );
    for (final match in importRegex.allMatches(code)) {
      final stmt = match.group(0)!.trim();
      final pkgMatch = RegExp(
        r'''^import\s+['"]package:([\w\d_\-]+)/''',
      ).firstMatch(stmt);
      final String name;
      if (pkgMatch != null) {
        name = pkgMatch.group(1)!;
      } else {
        final dartMatch = RegExp(
          r'''^import\s+['"]dart:([\w\d_\-]+)''',
        ).firstMatch(stmt);
        name = dartMatch?.group(1) ?? 'library';
      }
      imports.add(ParsedCellImport(statement: stmt, name: name));
    }
    final body = code.replaceAll(importRegex, '');
    return (imports: imports, body: body);
  }

  /// Splits [source] into top-level statements and declarations while respecting
  /// comments, string literals (including triple-quoted, raw, and interpolated
  /// strings), and nested `()`, `[]`, `{}` blocks.
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
      final nextAfterComment = _skipComment(source, i);
      if (nextAfterComment != i) {
        i = nextAfterComment;
        continue;
      }

      final nextAfterString = _skipStringLiteral(source, i);
      if (nextAfterString != i) {
        i = nextAfterString;
        continue;
      }

      final ch = source.codeUnitAt(i);
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
              (nextWord == 'while' &&
                  RegExp(
                    r'^(?:[A-Za-z_$][\w$]*\s*:\s*)?do\b',
                  ).hasMatch(currentText));
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
      r'^(?:[A-Za-z_$][\w$]*\s*:\s*)?(?:if|for|while|switch|try|await\s+for)\b',
    ).hasMatch(strippedText)) {
      return true;
    }
    return _matchFunctionDeclarationName(strippedText) != null;
  }

  static ParsedCellItem _classifyItem(String rawItem) {
    final trimmedItem = rawItem.trim();
    final stripped = _stripLeadingTrivia(trimmedItem).trim();

    String ensureStatementTerminated(String s) {
      if (!s.endsWith('}') && !s.endsWith(';')) {
        return '$s;';
      }
      return s;
    }

    // 1. class / enum / mixin / extension / typedef
    final typeMatch = RegExp(
      r'^(?:abstract\s+|base\s+|final\s+|interface\s+|sealed\s+|mixin\s+)*(?:class|enum|mixin|extension|typedef)\s+([A-Za-z_$][\w$]*)',
    ).firstMatch(stripped);
    if (typeMatch != null) {
      return ParsedCellItem(
        type: ParsedCellItemType.typeOrFunctionDeclaration,
        source: ensureStatementTerminated(trimmedItem),
        symbolName: typeMatch.group(1),
      );
    }
    if (RegExp(r'^extension\s+on\b').hasMatch(stripped)) {
      return ParsedCellItem(
        type: ParsedCellItemType.typeOrFunctionDeclaration,
        source: ensureStatementTerminated(trimmedItem),
      );
    }

    // 2. Top-level function declaration: [returnType] name([params]) [async] { or =>
    final fnName = _matchFunctionDeclarationName(stripped);
    if (fnName != null) {
      return ParsedCellItem(
        type: ParsedCellItemType.typeOrFunctionDeclaration,
        source: ensureStatementTerminated(trimmedItem),
        symbolName: fnName,
      );
    }

    // 3. Control-flow / jump / labeled / `await for` statement
    if (RegExp(
      r'^(?:[A-Za-z_$][\w$]*\s*:\s*)?await\s+for\b',
    ).hasMatch(stripped)) {
      return ParsedCellItem(
        type: ParsedCellItemType.statement,
        source: ensureStatementTerminated(trimmedItem),
      );
    }
    if (RegExp(
      r'^[A-Za-z_$][\w$]*\s*:\s*(?:for|while|do|switch|if|try)\b',
    ).hasMatch(stripped)) {
      return ParsedCellItem(
        type: ParsedCellItemType.statement,
        source: ensureStatementTerminated(trimmedItem),
      );
    }
    final firstWordMatch = RegExp(
      r'^([A-Za-z_$][\w$]*)\b',
    ).firstMatch(stripped);
    final firstWord = firstWordMatch?.group(1);
    if (firstWord != null &&
        _statementKeywords.contains(firstWord) &&
        firstWord != 'await') {
      return ParsedCellItem(
        type: ParsedCellItemType.statement,
        source: ensureStatementTerminated(trimmedItem),
      );
    }

    // 4. Variable declaration: (late)? (var|final|const|<Type>) name = ...
    final varDecl = _tryParseVariableDeclaration(stripped);
    if (varDecl != null && varDecl.variables.isNotEmpty) {
      return ParsedCellItem(
        type: ParsedCellItemType.variableDeclaration,
        source: _stripTrailingSemicolon(trimmedItem),
        isConst: varDecl.isConst,
        isFinal: varDecl.isFinal,
        typeAnnotation: varDecl.typeAnnotation,
        variables: varDecl.variables,
      );
    }

    // Pattern variable declarations such as `final (a, b) = (1, 2);` are
    // statements, not expressions.
    if (RegExp(r'^(?:late\s+)?(?:var|final|const)\b').hasMatch(stripped)) {
      return ParsedCellItem(
        type: ParsedCellItemType.statement,
        source: ensureStatementTerminated(trimmedItem),
      );
    }

    // 5. Otherwise it is an expression statement.
    return ParsedCellItem(
      type: ParsedCellItemType.expression,
      source: _stripTrailingSemicolon(trimmedItem),
    );
  }

  static String? _matchFunctionDeclarationName(String stripped) {
    // Either `[returnType] name[<TypeParams>](` or `name[<TypeParams>](`.
    String? candidateName;
    int? openParenIdx;

    if (_consumeTypeAnnotation(stripped) case final consumed?) {
      final rest = consumed.rest;
      final nameMatch = RegExp(r'^([A-Za-z_$][\w$]*)').firstMatch(rest);
      if (nameMatch != null) {
        final name = nameMatch.group(1)!;
        var idx = nameMatch.end;
        idx = _skipSpaces(rest, idx);
        if (idx < rest.length && rest.codeUnitAt(idx) == 0x3C /* < */ ) {
          final afterTypeParams = _skipBalancedAngles(rest, idx);
          if (afterTypeParams != -1) {
            idx = _skipSpaces(rest, afterTypeParams);
          }
        }
        if (idx < rest.length && rest.codeUnitAt(idx) == 0x28 /* ( */ ) {
          candidateName = name;
          openParenIdx = stripped.length - rest.length + idx;
        }
      }
    }

    if (candidateName == null) {
      final nameMatch = RegExp(r'^([A-Za-z_$][\w$]*)').firstMatch(stripped);
      if (nameMatch == null) return null;
      final name = nameMatch.group(1)!;
      var idx = nameMatch.end;
      idx = _skipSpaces(stripped, idx);
      if (idx < stripped.length && stripped.codeUnitAt(idx) == 0x3C /* < */ ) {
        final afterTypeParams = _skipBalancedAngles(stripped, idx);
        if (afterTypeParams != -1) {
          idx = _skipSpaces(stripped, afterTypeParams);
        }
      }
      if (idx < stripped.length && stripped.codeUnitAt(idx) == 0x28 /* ( */ ) {
        candidateName = name;
        openParenIdx = idx;
      }
    }

    if (candidateName == null || openParenIdx == null) return null;
    if (_statementKeywords.contains(candidateName) ||
        candidateName == 'var' ||
        candidateName == 'final' ||
        candidateName == 'const' ||
        candidateName == 'print' ||
        candidateName == 'display') {
      return null;
    }

    // Find matching ')' for the parameter list '(' while skipping strings/comments.
    var depth = 0;
    var closeParenIdx = -1;
    var i = openParenIdx;
    while (i < stripped.length) {
      final nextComment = _skipComment(stripped, i);
      if (nextComment != i) {
        i = nextComment;
        continue;
      }
      final nextStr = _skipStringLiteral(stripped, i);
      if (nextStr != i) {
        i = nextStr;
        continue;
      }
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
      i++;
    }
    if (closeParenIdx == -1) return null;
    final afterParams = stripped.substring(closeParenIdx + 1).trimLeft();
    if (RegExp(
      r'^(?:(?:async|sync)\s*\*?\s*)?(?:\{|=>)',
    ).hasMatch(afterParams)) {
      return candidateName;
    }
    return null;
  }

  static ({
    bool isConst,
    bool isFinal,
    String? typeAnnotation,
    List<ParsedVarDeclarator> variables,
  })?
  _tryParseVariableDeclaration(String stripped) {
    var rest = _stripTrailingSemicolon(stripped).trim();
    final lateMatch = RegExp(r'^late\s+').firstMatch(rest);
    if (lateMatch != null) {
      rest = rest.substring(lateMatch.end).trimLeft();
    }

    var isConst = false;
    var isFinal = false;
    String? typeAnnotation;
    String? declaratorsPart;

    final kwMatch = RegExp(r'^(var|final|const)\s+').firstMatch(rest);
    if (kwMatch != null) {
      final kw = kwMatch.group(1)!;
      isConst = kw == 'const';
      isFinal = kw == 'final';
      final afterKw = rest.substring(kwMatch.end).trimLeft();
      if (kw == 'var' ||
          RegExp(r'^[A-Za-z_$][\w$]*\s*(?:=|,|$)').hasMatch(afterKw)) {
        declaratorsPart = afterKw;
      } else if (_consumeTypeAnnotation(afterKw) case final consumed?) {
        typeAnnotation = consumed.typeAnnotation;
        declaratorsPart = consumed.rest;
      }
    } else if (_consumeTypeAnnotation(rest) case final consumed?) {
      typeAnnotation = consumed.typeAnnotation;
      declaratorsPart = consumed.rest;
    }

    if (declaratorsPart == null || declaratorsPart.isEmpty) return null;
    final trimmedDecls = declaratorsPart.trimLeft();
    if (trimmedDecls.startsWith('(') ||
        trimmedDecls.startsWith('[') ||
        trimmedDecls.startsWith('{')) {
      return null;
    }

    final parts = _splitTopLevelDeclaratorCommas(declaratorsPart);
    final variables = <ParsedVarDeclarator>[];
    for (final part in parts) {
      final trimmed = part.trim();
      final eqIdx = _findTopLevelEquals(trimmed);
      if (eqIdx == -1) {
        if (RegExp(r'^[A-Za-z_$][\w$]*$').hasMatch(trimmed)) {
          variables.add(ParsedVarDeclarator(name: trimmed));
        } else {
          return null;
        }
      } else {
        final lhs = trimmed.substring(0, eqIdx).trim();
        final rhs = trimmed.substring(eqIdx + 1).trim();
        if (!RegExp(r'^[A-Za-z_$][\w$]*$').hasMatch(lhs) || rhs.isEmpty) {
          return null;
        }
        variables.add(ParsedVarDeclarator(name: lhs, initializer: rhs));
      }
    }
    return (
      isConst: isConst,
      isFinal: isFinal,
      typeAnnotation: typeAnnotation,
      variables: variables,
    );
  }

  /// Attempts to consume a leading type annotation (such as `int`, `math.Point`,
  /// `List<int>`, or `Map<String, List<int>>?`) from [s], returning the
  /// consumed type annotation and the remaining text starting with an identifier.
  static ({String typeAnnotation, String rest})? _consumeTypeAnnotation(
    String s,
  ) {
    final headMatch = RegExp(
      r'^([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)',
    ).firstMatch(s);
    if (headMatch == null) return null;
    final fullHead = headMatch.group(1)!;
    final firstId = fullHead.split('.').first;
    if (_statementKeywords.contains(firstId) ||
        firstId == 'var' ||
        firstId == 'final' ||
        firstId == 'const' ||
        firstId == 'late') {
      return null;
    }

    var idx = headMatch.end;
    final afterHead = _skipSpaces(s, idx);
    if (afterHead < s.length && s.codeUnitAt(afterHead) == 0x3C /* < */ ) {
      final afterAngles = _skipBalancedAngles(s, afterHead);
      if (afterAngles == -1) return null;
      idx = afterAngles;
    }
    if (idx < s.length && s.codeUnitAt(idx) == 0x3F /* ? */ ) {
      idx++;
    }
    // Must be followed by whitespace and then an identifier start.
    if (idx >= s.length) return null;
    final afterType = _skipSpaces(s, idx);
    if (afterType == idx || afterType >= s.length) return null;
    if (!RegExp(r'^[A-Za-z_$]').hasMatch(s.substring(afterType))) {
      return null;
    }
    return (
      typeAnnotation: s.substring(0, idx).trim(),
      rest: s.substring(afterType),
    );
  }

  static int _skipSpaces(String s, int index) {
    var i = index;
    while (i < s.length) {
      final c = s.codeUnitAt(i);
      if (c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D) {
        i++;
      } else {
        break;
      }
    }
    return i;
  }

  /// Scans balanced `<...>` starting at `s[openAngleIndex] == '<'`, returning
  /// the index immediately after the matching `>`, or `-1` if unbalanced.
  static int _skipBalancedAngles(String s, int openAngleIndex) {
    var depth = 0;
    for (var i = openAngleIndex; i < s.length; i++) {
      final c = s.codeUnitAt(i);
      if (c == 0x3C /* < */ ) {
        depth++;
      } else if (c == 0x3E /* > */ ) {
        depth--;
        if (depth == 0) return i + 1;
      } else if (c == 0x3B /* ; */ || c == 0x7B /* { */ || c == 0x7D /* } */ ) {
        return -1;
      }
    }
    return -1;
  }

  /// Splits a variable declarator list `a = 1, b = 2` on top-level commas that
  /// precede a subsequent declarator identifier (`id (= | , | $)`), while
  /// skipping string literals, comments, `()`, `[]`, `{}`, and `<TypeA, TypeB>`
  /// type argument commas.
  static List<String> _splitTopLevelDeclaratorCommas(String s) {
    final parts = <String>[];
    var start = 0;
    var depth = 0;
    var i = 0;
    while (i < s.length) {
      final nextComment = _skipComment(s, i);
      if (nextComment != i) {
        i = nextComment;
        continue;
      }
      final nextStr = _skipStringLiteral(s, i);
      if (nextStr != i) {
        i = nextStr;
        continue;
      }
      final c = s.codeUnitAt(i);
      if (c == 0x28 || c == 0x5B || c == 0x7B) {
        depth++;
      } else if (c == 0x29 || c == 0x5D || c == 0x7D) {
        if (depth > 0) depth--;
      } else if (c == 0x2C /* , */ && depth == 0) {
        final afterComma = _stripLeadingTrivia(s.substring(i + 1));
        if (RegExp(r'^[A-Za-z_$][\w$]*\s*(?:=|,|$)').hasMatch(afterComma)) {
          parts.add(s.substring(start, i));
          start = i + 1;
        }
      }
      i++;
    }
    parts.add(s.substring(start));
    return parts;
  }

  static int _findTopLevelEquals(String s) {
    var depth = 0;
    var i = 0;
    while (i < s.length) {
      final nextComment = _skipComment(s, i);
      if (nextComment != i) {
        i = nextComment;
        continue;
      }
      final nextStr = _skipStringLiteral(s, i);
      if (nextStr != i) {
        i = nextStr;
        continue;
      }
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
      i++;
    }
    return -1;
  }

  /// If [s] at [index] begins a `//` or `/* ... */` comment, returns the index
  /// after the comment; otherwise returns [index].
  static int _skipComment(String s, int index) {
    final len = s.length;
    if (index + 1 >= len || s.codeUnitAt(index) != 0x2F /* / */ ) {
      return index;
    }
    final next = s.codeUnitAt(index + 1);
    if (next == 0x2F /* / */ ) {
      var i = index + 2;
      while (i < len && s.codeUnitAt(i) != 0x0A) {
        i++;
      }
      return i;
    }
    if (next == 0x2A /* * */ ) {
      var i = index + 2;
      while (i + 1 < len &&
          !(s.codeUnitAt(i) == 0x2A && s.codeUnitAt(i + 1) == 0x2F)) {
        i++;
      }
      return (i + 2 <= len) ? i + 2 : len;
    }
    return index;
  }

  /// If [s] at [index] begins a string literal (single, double, triple-quoted,
  /// or raw `r'...'`/`r"..."`), returns the index after the closing quote,
  /// recursively skipping `${...}` interpolations in non-raw strings.
  static int _skipStringLiteral(String s, int index) {
    final len = s.length;
    if (index >= len) return index;
    final ch = s.codeUnitAt(index);
    final isRaw =
        ch == 0x72 /* r */ &&
        index + 1 < len &&
        (s.codeUnitAt(index + 1) == 0x27 || s.codeUnitAt(index + 1) == 0x22);
    if (!isRaw && ch != 0x27 /* ' */ && ch != 0x22 /* " */ ) {
      return index;
    }

    var i = isRaw ? index + 1 : index;
    final quote = s.codeUnitAt(i);
    final isTriple =
        i + 2 < len &&
        s.codeUnitAt(i + 1) == quote &&
        s.codeUnitAt(i + 2) == quote;
    i += isTriple ? 3 : 1;

    while (i < len) {
      final c = s.codeUnitAt(i);
      if (!isRaw && c == 0x5C /* \ */ ) {
        i += 2;
        continue;
      }
      if (!isRaw &&
          c == 0x24 /* $ */ &&
          i + 1 < len &&
          s.codeUnitAt(i + 1) == 0x7B /* { */ ) {
        i += 2;
        var interpDepth = 1;
        while (i < len && interpDepth > 0) {
          final nextComment = _skipComment(s, i);
          if (nextComment != i) {
            i = nextComment;
            continue;
          }
          final nextStr = _skipStringLiteral(s, i);
          if (nextStr != i) {
            i = nextStr;
            continue;
          }
          final ic = s.codeUnitAt(i);
          if (ic == 0x7B /* { */ ) {
            interpDepth++;
          } else if (ic == 0x7D /* } */ ) {
            interpDepth--;
          }
          i++;
        }
        continue;
      }
      if (isTriple) {
        if (i + 2 < len &&
            s.codeUnitAt(i) == quote &&
            s.codeUnitAt(i + 1) == quote &&
            s.codeUnitAt(i + 2) == quote) {
          return i + 3;
        }
      } else if (c == quote) {
        return i + 1;
      }
      i++;
    }
    return len;
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
      final nextComment = _skipComment(s, i);
      if (nextComment != i) {
        i = nextComment;
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

  static String _escapeDartString(String s) => s
      .replaceAll(r'\', r'\\')
      .replaceAll("'", r"\'")
      .replaceAll(r'$', r'\$')
      .replaceAll('\n', r'\n')
      .replaceAll('\r', r'\r');
}
