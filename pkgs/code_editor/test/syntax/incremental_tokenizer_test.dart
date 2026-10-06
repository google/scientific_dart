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

import 'package:code_editor/syntax.dart';
import 'package:test/test.dart';

void main() {
  group('IncrementalTokenizer & DartGrammar Tests', () {
    test('Tokenizes line tokens with scope rules', () {
      final theme = ColorTheme.darkPlus();
      final cache = StyleCache(theme);

      final scopes = [StyleScope('keyword.control.dart')];
      final style = cache.getStyle(scopes);
      expect(style.foreground, equals(0xFF569CD6));
    });

    test(
      'Distinguishes keywords, types, functions, named args, and properties',
      () {
        final tokenizer = IncrementalTokenizer(
          tokenizer: DartGrammar.createLexer(),
          styleCache: StyleCache(ColorTheme.catppuccinMocha()),
        );
        tokenizer.setDocument([
          'final NDArray<Float64> x = linspace<Float64>(0.0, 1.0, 60, dtype: DType.float64);',
        ]);

        final tokens = tokenizer.getTokensForLine(0);
        SyntaxToken tokenFor(String text) =>
            tokens.firstWhere((t) => t.text == text);

        // 'final' -> storage.modifier.dart (Mauve 0xFFCBA6F7)
        expect(tokenFor('final').style.foreground, equals(0xFFCBA6F7));
        // 'NDArray' & 'Float64' -> entity.name.type.dart (Yellow 0xFFF9E2AF)
        expect(tokenFor('NDArray').style.foreground, equals(0xFFF9E2AF));
        expect(tokenFor('Float64').style.foreground, equals(0xFFF9E2AF));
        // 'x' -> variable.other.dart (Text 0xFFCDD6F4)
        expect(tokenFor('x').style.foreground, equals(0xFFCDD6F4));
        // 'linspace' -> entity.name.function.dart (Blue 0xFF89B4FA)
        expect(tokenFor('linspace').style.foreground, equals(0xFF89B4FA));
        // '0.0' -> constant.numeric.decimal.dart (Peach 0xFFFAB387)
        expect(tokenFor('0.0').style.foreground, equals(0xFFFAB387));
        // 'dtype' -> variable.parameter.named.dart (Maroon 0xFFEBA0AC, italic)
        expect(tokenFor('dtype').style.foreground, equals(0xFFEBA0AC));
        expect(tokenFor('dtype').style.italic, isTrue);
        // 'float64' after '.' -> variable.other.property.dart (Teal 0xFF94E2D5)
        expect(tokenFor('float64').style.foreground, equals(0xFF94E2D5));
      },
    );

    test('Tokenizes string interpolation, escapes, and raw strings', () {
      final tokenizer = IncrementalTokenizer(
        tokenizer: DartGrammar.createLexer(),
        styleCache: StyleCache(ColorTheme.catppuccinMocha()),
      );
      tokenizer.setDocument([
        r"'Value $x and ${a + 1}\n'",
        r"r'\int_0^\infty $notInterpolated'",
      ]);

      final interpTokens = tokenizer.getTokensForLine(0);
      SyntaxToken tokenInLine0(String text) =>
          interpTokens.firstWhere((t) => t.text == text);

      expect(tokenInLine0(r'$x').style.foreground, equals(0xFFF38BA8));
      expect(tokenInLine0(r'${').style.foreground, equals(0xFFF38BA8));
      expect(tokenInLine0('a').style.foreground, equals(0xFFCDD6F4));
      expect(tokenInLine0('+').style.foreground, equals(0xFF89DCEB));
      expect(tokenInLine0('1').style.foreground, equals(0xFFFAB387));
      expect(tokenInLine0('}').style.foreground, equals(0xFFF38BA8));
      expect(tokenInLine0(r'\n').style.foreground, equals(0xFFF5C2E7));

      // Raw string is a single unsplit string token
      final rawTokens = tokenizer.getTokensForLine(1);
      expect(rawTokens.length, equals(1));
      expect(rawTokens.first.type, equals(TokenType.string));
      expect(rawTokens.first.style.foreground, equals(0xFFA6E3A1));
    });

    test('Tokenizes doc comment references and inline code', () {
      final tokenizer = IncrementalTokenizer(
        tokenizer: DartGrammar.createLexer(),
        styleCache: StyleCache(ColorTheme.catppuccinMocha()),
      );
      tokenizer.setDocument(['/// Computes [NDArray] with `linspace`.']);

      final tokens = tokenizer.getTokensForLine(0);
      SyntaxToken tokenFor(String text) =>
          tokens.firstWhere((t) => t.text == text);

      expect(tokenFor('[NDArray]').style.foreground, equals(0xFF89B4FA));
      expect(tokenFor('`linspace`').style.foreground, equals(0xFFA6E3A1));
    });
  });
}
