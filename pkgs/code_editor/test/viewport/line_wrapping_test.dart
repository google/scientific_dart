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

import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('LineWrappingEngine', () {
    test('computes word-boundary soft wraps correctly', () {
      const engine = LineWrappingEngine(maxColumns: 10, enabled: true);
      final slices = engine.wrapLine('hello world flutter dart', 0);

      expect(slices.length, greaterThan(1));
      expect(slices.first.content, equals('hello '));
    });

    test('wraps long lines into multiple slices', () {
      const engine = LineWrappingEngine(maxColumns: 10, enabled: true);
      final slices = engine.wrapLine('abcdefghijklmnopqrstuvwxyz', 0);

      expect(slices.length, equals(3));
      expect(slices[0].content, equals('abcdefghij'));
      expect(slices[1].content, equals('klmnopqrst'));
      expect(slices[2].content, equals('uvwxyz'));
    });
  });
}
