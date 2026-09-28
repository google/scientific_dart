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

import 'package:test/test.dart';
import 'package:notebook/notebook.dart';
import 'package:notebook/src/kernel_helper.dart';

void main() {
  test('LaTeX and Latex generate valid math-latex HTML markup', () {
    final eq1 = LaTeX(r'E = m c^2');
    expect(eq1.toHtml(), contains('class="math-latex"'));
    expect(eq1.toHtml(), contains(r'\(E = m c^2\)'));

    final eq2 = Latex(r'A x = \lambda x');
    expect(prettyFormat(eq2), contains(r'\(A x = \lambda x\)'));
  });
}
