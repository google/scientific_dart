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
import 'package:test/test.dart';

void main() {
  group('DartFormatterEngine Tests', () {
    test('formats messy Dart loops and braces with consistent indentation', () {
      const input = '''
void main(){
int a=10;
for(var i=0;i<a;i++){
print(i);
}
}
''';

      const formatter = DartFormatterEngine(tabSize: 2);
      final formatted = formatter.formatCode(input);

      expect(formatted, '''
void main() {
  int a = 10;
  for (var i = 0; i < a; i++) {
    print(i);
  }
}''');
    });

    test('formats comparison and assignment operators cleanly', () {
      const input = 'final x=10+20*3;if(x>=50){x+=5;}';
      const formatter = DartFormatterEngine(tabSize: 2);
      final formatted = formatter.formatCode(input);

      expect(formatted, '''
final x = 10+20*3;
if (x >= 50) {
  x += 5;
}''');
    });

    test('preserves verbatim single line and doc comments', () {
      const input = '''
/// Creates an NDArray.
void create(){
  // initialize
  var x=1;
}
''';
      const formatter = DartFormatterEngine(tabSize: 2);
      final formatted = formatter.formatCode(input);

      expect(formatted, '''
/// Creates an NDArray.
void create() {
  // initialize
  var x = 1;
}''');
    });

    test(
      'preserves generic type arguments without spaces around angle brackets',
      () {
        const input =
            'var x = linspace<Float64>(Float64(0.0), Float64(10.0), 100);';
        const formatter = DartFormatterEngine(tabSize: 2);
        final formatted = formatter.formatCode(input);

        expect(
          formatted,
          'var x = linspace<Float64>(Float64(0.0), Float64(10.0), 100);',
        );
      },
    );

    test(
      'does not corrupt ==, !=, <=, >=, or => operators or string literals',
      () {
        const input = '''
final eq = a == b && c != d && e <= f && g >= h;
final fn = (int x) => x == 0 ? "a=b" : 'c==d';
''';
        const formatter = DartFormatterEngine(tabSize: 2);
        final formatted = formatter.formatCode(input);

        expect(formatted, '''
final eq = a == b && c != d && e <= f && g >= h;
final fn = (int x) => x == 0 ? "a=b" : 'c==d';''');
      },
    );

    test('delegates to customFormatter when provided', () {
      final formatter = DartFormatterEngine(
        customFormatter: (code) => '// custom\n${code.trim()}',
      );
      expect(formatter.formatCode('var x=1;'), '// custom\nvar x=1;');
    });
  });
}
