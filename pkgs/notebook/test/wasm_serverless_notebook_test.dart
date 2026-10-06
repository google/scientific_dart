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

import 'dart:convert';
import 'dart:io';

import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/diagnostic/diagnostic.dart';
import 'package:notebook/notebook.dart';
import 'package:path/path.dart' as p;
import 'package:puppeteer/puppeteer.dart';
import 'package:test/test.dart';

String _findRepoRoot() {
  var dir = Directory.current.absolute;
  while (true) {
    if (File(p.join(dir.path, 'tool', 'build_wasm.dart')).existsSync()) {
      return dir.path;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError(
        'Could not locate workspace root from ${Directory.current.path}',
      );
    }
    dir = parent;
  }
}

String? _findChromeExecutable() {
  for (final candidate in const [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ]) {
    if (File(candidate).existsSync()) {
      return candidate;
    }
  }
  return null;
}

void main() {
  group('WasmCellBundler unit tests', () {
    test(
      'bundles single-variable declaration and trailing expression across cells',
      () {
        final cells = [
          const WasmNotebookCell(
            id: 'c1',
            code:
                'var a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);',
          ),
          const WasmNotebookCell(id: 'c2', code: 'a * 2'),
        ];

        final bundled = WasmCellBundler.bundleCells(cells);
        expect(bundled.includedCellIds, ['c1', 'c2']);
        expect(bundled.declaredVariables, ['a']);
        expect(
          bundled.mainDartSource,
          contains(
            'var a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);',
          ),
        );
        expect(
          bundled.mainDartSource,
          contains("Declared variable a\\nValue:"),
        );
        expect(
          bundled.mainDartSource,
          contains(
            '_cellValue = await _evaluateCellExpression(() async => (\na * 2\n));',
          ),
        );
      },
    );

    test(
      'hoists top-level functions and classes and rewrites kernel_helper import',
      () {
        final cells = [
          const WasmNotebookCell(
            id: 'c1',
            code: '''
import 'package:notebook/src/kernel_helper.dart';
import 'package:pocketfft/pocketfft.dart';

int doubleIt(int x) => x * 2;

final y = doubleIt(21);
display(Plot(y: NDArray.fromList([1.0, 2.0], [2], DType.float64)))
''',
          ),
        ];

        final bundled = WasmCellBundler.bundleCells(cells);
        expect(
          bundled.mainDartSource,
          contains("import 'package:notebook/src/notebook_widgets.dart';"),
        );
        expect(
          bundled.mainDartSource,
          isNot(contains('package:notebook/src/kernel_helper.dart')),
        );
        expect(
          bundled.mainDartSource,
          contains("import 'package:pocketfft/pocketfft.dart';"),
        );
        expect(
          bundled.mainDartSource,
          contains('int doubleIt(int x) => x * 2;'),
        );
        expect(bundled.declaredVariables, ['y']);
        expect(
          bundled.mainDartSource,
          contains(
            '_cellValue = await _evaluateCellExpression(() async => (\n'
            'display(Plot(y: NDArray.fromList([1.0, 2.0], [2], DType.float64)))\n'
            '));',
          ),
        );
      },
    );

    test('captures a trailing void expression so the cell runs for effect', () {
      final cells = [
        const WasmNotebookCell(id: 'c1', code: 'final xs = [1, 2];'),
        const WasmNotebookCell(id: 'c2', code: 'xs.forEach(print)'),
      ];

      final bundled = WasmCellBundler.bundleCells(cells);
      expect(
        bundled.mainDartSource,
        contains(
          'Future<dynamic> _evaluateCellExpression(Future<dynamic> Function() thunk)',
        ),
      );
      expect(
        bundled.mainDartSource,
        contains(
          '_cellValue = await _evaluateCellExpression(() async => (\n'
          'xs.forEach(print)\n'
          '));',
        ),
      );
    });

    test(
      'generated program has no static errors for void trailing expressions',
      () async {
        final cells = [
          const WasmNotebookCell(id: 'c1', code: "print('hi')"),
          const WasmNotebookCell(
            id: 'c2',
            code: 'final xs = [1, 2];\nxs.forEach(print)',
          ),
          const WasmNotebookCell(id: 'c3', code: "display(Html('<b>x</b>'))"),
          const WasmNotebookCell(id: 'c4', code: 'xs.add(3)'),
          const WasmNotebookCell(id: 'c5', code: 'xs.length'),
          const WasmNotebookCell(id: 'c6', code: 'await Future<int>.value(3)'),
        ];
        final bundled = WasmCellBundler.bundleCells(cells);

        // The program is written inside the package so that its `package:`
        // imports resolve through the workspace package config.
        final scratchDir = Directory(
          p.join(
            _findRepoRoot(),
            'pkgs',
            'notebook',
            '.dart_tool',
            'wasm_cell_bundler_test',
          ),
        )..createSync(recursive: true);
        final programDir = scratchDir.createTempSync();
        addTearDown(() => programDir.deleteSync(recursive: true));
        final mainFile = File(p.join(programDir.path, 'cell_main.dart'))
          ..writeAsStringSync(bundled.mainDartSource);

        final collection = AnalysisContextCollection(
          includedPaths: [mainFile.path],
        );
        addTearDown(collection.dispose);
        final result = await collection
            .contextFor(mainFile.path)
            .currentSession
            .getErrors(mainFile.path);
        final errors = (result as ErrorsResult).diagnostics
            .where((d) => d.severity == Severity.error)
            .map((d) => '${d.diagnosticCode.lowerCaseName}: ${d.message}')
            .toList();
        expect(errors, isEmpty);
      },
    );

    test('maps cursor offset accurately in bundleForAnalysis', () {
      const cell1Code = 'final a = NDArray.zeros([2, 2]);';
      const cell2Code = 'final b = a.resh';
      final cells = [
        const WasmNotebookCell(id: 'c1', code: cell1Code),
        const WasmNotebookCell(id: 'c2', code: cell2Code),
      ];

      final analysis = WasmCellBundler.bundleForAnalysis(
        cells,
        activeCellId: 'c2',
        cursorOffsetInCell: cell2Code.length,
      );
      expect(
        analysis.source.substring(
          analysis.mappedOffset - 'a.resh'.length,
          analysis.mappedOffset,
        ),
        'a.resh',
      );
    });

    test('ParsedNotebookCell aligns Wasm bundler and VM kernel transformation', () {
      // 1. Single variable declaration with await.
      final asyncVar = ParsedNotebookCell.parse('final x = await seven();');
      expect(asyncVar.singleDeclaredVariable?.name, 'x');
      final asyncVarTx = asyncVar.transformForKernel();
      expect(asyncVarTx.declaredVariables, ['x']);
      expect(
        asyncVarTx.cellBodyCode,
        contains('final x = __set_x(await seven());'),
      );
      expect(asyncVarTx.cellBodyCode, contains('Declared variable x'));

      // 2. Cell mixing a class declaration, typed variable with generics, and trailing expression.
      final mixed = ParsedNotebookCell.parse('''
class Box {
  final int v;
  Box(this.v);
}
final Map<String, int> m = <String, int>{'a': 21};
Box(m['a']!).v * 2
''');
      final mixedTx = mixed.transformForKernel();
      expect(mixedTx.namedDefinitions.keys, containsAll(['Box', 'm']));
      expect(mixedTx.declaredVariables, ['m']);
      expect(
        mixedTx.namedDefinitions['m'],
        contains('Map<String, int> get m => __slot_m as Map<String, int>;'),
      );
      expect(
        mixedTx.cellBodyCode,
        contains(
          "return await evaluateCellExpression(() async => (\nBox(m['a']!).v * 2\n));",
        ),
      );

      // 3. Function declaration with simple return type (`void`) followed by a call.
      final fnAndCall = ParsedNotebookCell.parse(
        "void greet() {\n  print('hello');\n}\ngreet()",
      );
      final fnTx = fnAndCall.transformForKernel();
      expect(fnTx.namedDefinitions.keys, ['greet']);
      expect(
        fnTx.cellBodyCode,
        contains(
          'return await evaluateCellExpression(() async => (\ngreet()\n));',
        ),
      );
    });
  });

  group('formatNotebookCellCode unit tests', () {
    test('formats single-variable declaration with and without semicolon', () {
      expect(
        formatNotebookCellCode(
          'var a=NDArray.fromList([1.0,2.0,3.0,4.0],[2,2],DType.float64);',
        ),
        'var a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);',
      );
      expect(
        formatNotebookCellCode(
          'var a=NDArray.fromList([1.0,2.0,3.0,4.0],[2,2],DType.float64)',
        ),
        'var a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64)',
      );
    });

    test('formats standalone expression without trailing semicolon', () {
      expect(formatNotebookCellCode('a*2+1'), 'a * 2 + 1');
      expect(
        formatNotebookCellCode('// Multiply every element by 2\na*2'),
        '// Multiply every element by 2\na * 2',
      );
    });

    test('formats multi-statement cell with trailing expression', () {
      const input = '''
final v=NDArray.arange(0.0,5.0);
for(var i=0;i<3;i++){
print(i);
}
sum(v)*2
''';
      expect(formatNotebookCellCode(input), '''
final v = NDArray.arange(0.0, 5.0);
for (var i = 0; i < 3; i++) {
  print(i);
}
sum(v) * 2''');
    });

    test('formats cell mixing imports, class declaration, and statements', () {
      const input = '''
import 'dart:math' as math;

class Point{final double x;Point(this.x);}

final p=Point(math.pi);
p.x*2
''';
      expect(formatNotebookCellCode(input), '''
import 'dart:math' as math;

class Point {
  final double x;
  Point(this.x);
}

final p = Point(math.pi);
p.x * 2''');
    });
  });

  group('Serverless Wasm Notebook E2E (no active backend)', () {
    late String repoRoot;
    late WasmNotebookBundleResult bundleResult;
    late NotebookServer server;
    Browser? browser;

    setUpAll(() async {
      repoRoot = _findRepoRoot();
      final sdkPath =
          Platform.environment['DART_SDK'] ??
          p.dirname(p.dirname(Platform.resolvedExecutable));
      final builder = WasmNotebookBuilder(
        workspaceRoot: repoRoot,
        dartSdkPath: sdkPath,
      );
      bundleResult = await builder.buildStaticBundle();

      server = NotebookServer(
        workspaceDir: p.join(repoRoot, 'pkgs', 'notebook'),
        dartSdkPath: sdkPath,
        port: 0,
        serverlessWasm: true,
        staticBundleDir: bundleResult.outputDir,
      );
      await server.start();
    });

    tearDownAll(() async {
      await browser?.close();
      await server.stop();
    });

    test(
      'runs ndarray cells, plots, variable inspector, completions, and ipynb import/export completely in the browser without /ws or /api',
      () async {
        // 1. Verify that /ws and /api/export/ipynb return 404 (no active backend).
        final httpClient = HttpClient();
        try {
          final wsReq = await httpClient.getUrl(
            Uri.parse('http://localhost:${server.actualPort}/ws'),
          );
          final wsRes = await wsReq.close();
          expect(wsRes.statusCode, HttpStatus.notFound);

          final apiReq = await httpClient.getUrl(
            Uri.parse('http://localhost:${server.actualPort}/api/export/ipynb'),
          );
          final apiRes = await apiReq.close();
          expect(apiRes.statusCode, HttpStatus.notFound);
        } finally {
          httpClient.close();
        }

        final chromePath = _findChromeExecutable();
        if (chromePath == null) {
          markTestSkipped('Chrome executable not found on this machine.');
          return;
        }

        browser = await puppeteer.launch(
          executablePath: chromePath,
          headless: true,
          args: ['--no-sandbox', '--disable-setuid-sandbox'],
        );
        final page = await browser!.newPage();

        final pageErrors = <String>[];
        final consoleLogs = <String>[];
        page.onConsole.listen((msg) {
          consoleLogs.add('[${msg.type}] ${msg.text}');
          if (msg.type == ConsoleMessageType.error) {
            pageErrors.add('CONSOLE ERROR: ${msg.text}');
          }
        });
        page.onError.listen((err) {
          pageErrors.add('PAGE ERROR: $err');
        });

        try {
          await page.goto(
            'http://localhost:${server.actualPort}/?mode=wasm',
            wait: Until.networkIdle,
          );

          // 2. Wait for WasmNotebookRuntime to finish initializing compiler_worker.wasm + native_math.wasm.
          await page.waitForFunction('''() => {
              const el = document.getElementById('statusText');
              return el && el.innerText.includes('Wasm Ready');
            }''', timeout: const Duration(seconds: 30));

          final statusText = await page.evaluate<String>(
            "() => document.getElementById('statusText').innerText",
          );
          expect(statusText, contains('Wasm Ready (Serverless)'));

          // 3. Run all default cells (Cell 1 defines 2x2 NDArray `a`, Cell 2 evaluates `a * 2`).
          await page.evaluate('() => window.runAllCells()');

          await page.waitForFunction('''() => {
              const outputs = Array.from(document.querySelectorAll('.cell[data-cell-type="code"] .output-container'));
              return outputs.length >= 2 &&
                     outputs[0].innerText.trim().length > 0 &&
                     !outputs[0].innerText.includes('Running...') &&
                     outputs[1].innerText.trim().length > 0 &&
                     !outputs[1].innerText.includes('Running...');
            }''', timeout: const Duration(seconds: 30));

          final cellOutputs = (await page.evaluate<List<dynamic>>(
            '() => Array.from(document.querySelectorAll(\'.cell[data-cell-type="code"] .output-container\')).map(e => e.innerText)',
          )).cast<String>();
          expect(cellOutputs[0], contains('Declared variable a'));
          expect(cellOutputs[0], contains('[[1., 2.],'));
          expect(cellOutputs[0], contains('[3., 4.]]'));
          expect(cellOutputs[1], contains('[[2., 4.],'));
          expect(cellOutputs[1], contains('[6., 8.]]'));

          // 4. Verify Variable Inspector shows `a` with shape [2, 2] and dtype float64.
          final inspectorText = await page.evaluate<String>(
            "() => document.getElementById('inspectorList').innerText",
          );
          expect(inspectorText, contains('a'));
          expect(inspectorText, contains('[NDArray]'));
          expect(
            inspectorText,
            contains('NDArray shape: [2, 2] | dtype: float64'),
          );

          // 5. Add a 3rd code cell that renders an SVG Plot and returns `sum(a)`.
          final cell3Id = await page.evaluate<String>('''() => {
            const id = window.addCell(
              "display(Plot(y: a.reshape([4]), title: 'Serverless Wasm Plot'));\\nsum(a)"
            );
            window.runCell(id);
            return id;
          }''');

          await page.waitForFunction(
            '''(cellId) => {
              const out = document.getElementById('output-' + cellId);
              return out && out.innerText.trim().length > 0 && !out.innerText.includes('Running...');
            }''',
            args: [cell3Id],
            timeout: const Duration(seconds: 30),
          );

          final cell3Html = await page.evaluate<String>(
            '(cellId) => document.getElementById("output-" + cellId).innerHTML',
            args: [cell3Id],
          );
          expect(cell3Html, contains('<svg'));
          expect(cell3Html, contains('Serverless Wasm Plot'));
          expect(cell3Html, contains('10.'));

          // 6. Re-run Cell 3 without changes and verify cached compilation succeeds immediately (compileMs == 0).
          final cachedCompileMs = await page.evaluate<int>(
            '''async (cellId) => {
              const res = await window.wasmRuntime.executeCells(window.collectNotebookCells(), cellId);
              return res && res.ok && res.cached ? res.compileMs : -1;
            }''',
            args: [cell3Id],
          );
          expect(cachedCompileMs, 0);

          // 7. Test completions and hover via window.wasmRuntime.
          final completionCount = await page.evaluate<int>(
            '''async (cellId) => {
            const items = await window.wasmRuntime.getCompletions(window.collectNotebookCells(), cellId, 'NDArray.', 8);
            return Array.isArray(items) ? items.length : 0;
          }''',
            args: [cell3Id],
          );
          expect(completionCount, greaterThan(0));

          final hoverResult = await page.evaluate<String>('''async () => {
            const res = await window.wasmRuntime.getHover('NDArray.zeros([2]);', 2);
            return res ? JSON.stringify(res) : '';
          }''');
          expect(hoverResult, contains('NDArray'));

          // 8. Test client-side .ipynb import in Serverless Wasm mode.
          final importedNb = IpynbNotebook(
            cells: [
              IpynbCell(
                id: 'wasm-imported-1',
                cellType: IpynbCellType.code,
                source: 'final v = NDArray.arange(0.0, 5.0);\nsum(v)',
              ),
            ],
          );
          await page.evaluate(
            '(content) => window.importIpynbContentClientSide(content)',
            args: [jsonEncode(importedNb.toJson())],
          );

          final importedCellCount = await page.evaluate<int>(
            "() => document.querySelectorAll('.cell').length",
          );
          expect(importedCellCount, 1);

          // 9. Test cell formatting (✨ Format button / window.formatCell) in Serverless Wasm mode
          // across both CodeMirror 5 and Custom Wasm DartEditor engines.
          final formattedInCm = await page.evaluate<String>('''async () => {
            if (window.__editorClientReadyPromise) {
              await window.__editorClientReadyPromise;
            }
            const id = window.addCell("final v=NDArray.arange(0.0,5.0);\\nsum(v)*2");
            window.formatCell(id);
            const cell = document.getElementById(id);
            return cell && cell._cm ? cell._cm.getValue() : '';
          }''');
          expect(
            formattedInCm,
            'final v = NDArray.arange(0.0, 5.0);\nsum(v) * 2',
          );

          final formattedInDartEditor = await page.evaluate<String>(
            '''async () => {
            window.switchEditorEngine('dart_editor');
            const id = window.addCell("var x=1+2;\\nx*3");
            window.formatCell(id);
            const cell = document.getElementById(id);
            const val = cell && cell._cm ? cell._cm.getValue() : '';
            window.switchEditorEngine('codemirror');
            return val;
          }''',
          );
          expect(formattedInDartEditor, 'var x = 1 + 2;\nx * 3');

          expect(pageErrors, isEmpty);
        } catch (e) {
          final currentStatus = await page.evaluate<String>(
            "() => document.getElementById('statusText')?.innerText ?? '<missing>'",
          );
          final currentOutputs = await page.evaluate<List<dynamic>>(
            "() => Array.from(document.querySelectorAll('.output-container')).map(e => e.innerText)",
          );
          print('--- DIAGNOSTIC DUMP ---');
          print('STATUS TEXT: $currentStatus');
          print('CELL OUTPUTS: $currentOutputs');
          print('CONSOLE LOGS:\n${consoleLogs.join('\n')}');
          print('PAGE ERRORS:\n${pageErrors.join('\n')}');
          rethrow;
        }
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );
  });
}
