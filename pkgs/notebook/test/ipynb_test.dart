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

import 'package:notebook/src/ipynb.dart';
import 'package:test/test.dart';

void main() {
  group('IpynbNotebook serialization and deserialization', () {
    test('Round-trips empty notebook', () {
      final nb = IpynbNotebook(cells: []);
      final jsonMap = nb.toJson();
      expect(jsonMap['nbformat'], 4);
      expect(jsonMap['nbformat_minor'], 5);
      expect(jsonMap['cells'], isEmpty);
      expect(jsonMap['metadata']['kernelspec']['language'], 'dart');

      final decoded = IpynbNotebook.fromJson(jsonMap);
      expect(decoded.cells, isEmpty);
      expect(decoded.metadata['kernelspec']['name'], 'dart');
    });

    test(
      'Encodes and decodes markdown and code cells with multi-line sources',
      () {
        final nb = IpynbNotebook(
          cells: [
            IpynbCell(
              id: 'cell-md-1',
              cellType: IpynbCellType.markdown,
              source: '# Introduction\nHere is some math:\n\$\$x^2 + y^2\$\$',
            ),
            IpynbCell(
              id: 'cell-code-1',
              cellType: IpynbCellType.code,
              source: 'final x = Symbol(\'x\');\nprint(x);\nx.pow(2)',
              executionCount: 1,
              outputs: [
                IpynbOutput.stream(text: 'x\n'),
                IpynbOutput.executeResult(
                  executionCount: 1,
                  data: {
                    'text/plain': 'x^2',
                    'text/html': '<span class="katex">x^2</span>',
                  },
                ),
              ],
            ),
          ],
        );

        final jsonStr = nb.toJsonString();
        final decoded = IpynbNotebook.fromJsonString(jsonStr);

        expect(decoded.cells.length, 2);
        expect(decoded.cells[0].cellType, IpynbCellType.markdown);
        expect(
          decoded.cells[0].source,
          '# Introduction\nHere is some math:\n\$\$x^2 + y^2\$\$',
        );

        expect(decoded.cells[1].cellType, IpynbCellType.code);
        expect(decoded.cells[1].executionCount, 1);
        expect(decoded.cells[1].outputs.length, 2);

        final streamOut = decoded.cells[1].outputs[0];
        expect(streamOut.outputType, IpynbOutputType.stream);
        expect(streamOut.name, 'stdout');
        expect(streamOut.text, 'x\n');

        final execOut = decoded.cells[1].outputs[1];
        expect(execOut.outputType, IpynbOutputType.executeResult);
        expect(execOut.data!['text/plain'], 'x^2');
        expect(execOut.data!['text/html'], '<span class="katex">x^2</span>');
      },
    );

    test('Converts from and to legacy session cells list', () {
      final sessionCells = [
        {
          'id': 'cell-1',
          'type': 'markdown',
          'code': '### Heading',
          'output': '',
          'isError': false,
          'evaluated': false,
        },
        {
          'id': 'cell-2',
          'type': 'code',
          'code': 'var a = [1, 2, 3];\nprint(a);',
          'output': '{"mimeType":"text/html","data":"<div>Table</div>"}',
          'outputs': [
            {'mimeType': 'text/html', 'data': '<div>Table</div>'},
          ],
          'isError': false,
          'evaluated': true,
        },
        {
          'id': 'cell-3',
          'type': 'code',
          'code': 'throw Exception("Crash");',
          'output': 'Exception: Crash',
          'isError': true,
          'evaluated': false,
        },
      ];

      final nb = IpynbNotebook.fromSessionCells(sessionCells);
      expect(nb.cells.length, 3);
      expect(nb.cells[0].cellType, IpynbCellType.markdown);
      expect(nb.cells[1].cellType, IpynbCellType.code);
      expect(nb.cells[1].outputs.length, 1);
      expect(nb.cells[1].outputs[0].data!['text/html'], '<div>Table</div>');
      expect(nb.cells[2].outputs.length, 1);
      expect(nb.cells[2].outputs[0].outputType, IpynbOutputType.error);
      expect(nb.cells[2].outputs[0].evalue, 'Exception: Crash');

      final exportedSession = nb.toSessionCells();
      expect(exportedSession.length, 3);
      expect(exportedSession[0]['type'], 'markdown');
      expect(exportedSession[0]['code'], '### Heading');
      expect(exportedSession[1]['type'], 'code');
      expect(exportedSession[1]['evaluated'], true);
      expect(exportedSession[1]['isError'], false);
      expect(exportedSession[2]['isError'], true);
    });

    test('Parses external standard Jupyter Notebook format', () {
      const rawJupyterJson = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Standard Jupyter\\n",
    "Sample markdown"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Hello World\\n"
     ]
    }
   ],
   "source": [
    "print('Hello World')\\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "dart"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
''';

      final nb = IpynbNotebook.fromJsonString(rawJupyterJson);
      expect(nb.cells.length, 2);
      expect(nb.cells[0].source, '# Standard Jupyter\nSample markdown');
      expect(nb.cells[1].executionCount, 42);
      expect(nb.cells[1].outputs.first.text, 'Hello World\n');

      final session = nb.toSessionCells();
      expect(session.length, 2);
      expect(session[0]['code'], '# Standard Jupyter\nSample markdown');
      expect(session[1]['output'], 'Hello World\n');
    });
  });
}
