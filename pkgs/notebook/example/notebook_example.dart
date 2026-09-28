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

import 'package:notebook/notebook.dart';

void main() {
  print('=== Jupyter Notebook (ipynb) Creation Example ===');

  // 1. Create notebook cells
  final markdownCell = IpynbCell(
    id: 'intro-cell',
    cellType: IpynbCellType.markdown,
    source:
        '# Scientific Dart Notebook\nDemonstrating notebook file generation.',
  );

  final codeCell = IpynbCell(
    id: 'code-cell-1',
    cellType: IpynbCellType.code,
    source: 'var a = 21;\nprint("The answer is \${a * 2}");',
    outputs: [IpynbOutput.stream(text: 'The answer is 42\n')],
    executionCount: 1,
  );

  // 2. Build the notebook document
  final notebook = IpynbNotebook(cells: [markdownCell, codeCell]);

  print('Notebook created with ${notebook.cells.length} cells.');
  print('\nJSON representation:\n${notebook.toJsonString(pretty: true)}');
}
