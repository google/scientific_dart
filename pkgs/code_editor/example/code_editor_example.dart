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

import 'package:code_editor/code_editor.dart';

void main() {
  print('=== CodeEditorController Example ===');

  // 1. Initialize controller with sample Dart code
  final controller = CodeEditorController(
    initialText: 'void main() {\n  print("Hello, scientific Dart!");\n}',
  );

  print('Lines count: ${controller.lineCount}');
  print('Content:\n${controller.text}\n');

  // 2. Inspect syntax tokens for line 0
  final tokens = controller.lineTokens.first;
  print('Tokens in line 0:');
  for (final token in tokens) {
    print(
      '  ${token.type}: "${token.text}" (offset ${token.offset}..${token.end})',
    );
  }

  // 3. Perform text manipulation
  controller.selection = const TextSelection.collapsed(TextPosition(1, 2));
  controller.insertText('// Added by controller\n  ');
  print('\nUpdated Content:\n${controller.text}');
}
