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
  group('Standard TextMate Themes', () {
    test('Dark+ theme resolves standard tokens correctly', () {
      final theme = ColorTheme.darkPlus();
      expect(theme.name, equals('Dark+'));
      expect(theme.defaultStyle.foreground, equals(0xFFD4D4D4));
      expect(theme.defaultStyle.background, equals(0xFF1E1E1E));

      final commentStyle = theme.resolveStyle([
        StyleScope('comment.line.double.dart'),
      ]);
      expect(commentStyle.foreground, equals(0xFF6A9955));
      expect(commentStyle.italic, isTrue);

      final stringStyle = theme.resolveStyle([
        StyleScope('string.quoted.double.dart'),
      ]);
      expect(stringStyle.foreground, equals(0xFFCE9178));

      final keywordStyle = theme.resolveStyle([
        StyleScope('keyword.control.dart'),
      ]);
      expect(keywordStyle.foreground, equals(0xFF569CD6));

      final funcStyle = theme.resolveStyle([
        StyleScope('entity.name.function.dart'),
      ]);
      expect(funcStyle.foreground, equals(0xFFDCDCAA));
    });

    test('Light+ theme resolves standard tokens correctly', () {
      final theme = ColorTheme.lightPlus();
      expect(theme.name, equals('Light+'));
      expect(theme.defaultStyle.foreground, equals(0xFF000000));
      expect(theme.defaultStyle.background, equals(0xFFFFFFFF));

      final commentStyle = theme.resolveStyle([StyleScope('comment')]);
      expect(commentStyle.foreground, equals(0xFF008000));

      final stringStyle = theme.resolveStyle([StyleScope('string')]);
      expect(stringStyle.foreground, equals(0xFFA31515));
    });

    test('Monokai theme resolves standard tokens correctly', () {
      final theme = ColorTheme.monokai();
      expect(theme.name, equals('Monokai'));
      expect(theme.defaultStyle.foreground, equals(0xFFF8F8F2));
      expect(theme.defaultStyle.background, equals(0xFF272822));

      final keywordStyle = theme.resolveStyle([StyleScope('keyword')]);
      expect(keywordStyle.foreground, equals(0xFFF92672));
    });

    test('Solarized Dark theme resolves standard tokens correctly', () {
      final theme = ColorTheme.solarizedDark();
      expect(theme.name, equals('Solarized Dark'));
      expect(theme.defaultStyle.foreground, equals(0xFF839496));
      expect(theme.defaultStyle.background, equals(0xFF002B36));

      final stringStyle = theme.resolveStyle([StyleScope('string')]);
      expect(stringStyle.foreground, equals(0xFF2AA198));
    });
  });

  group('VS Code Theme JSON Parser & Hex Colors', () {
    test('parseHexColor handles 3, 6, and 8 digit hex formats', () {
      expect(parseHexColor('#FFF'), equals(0xFFFFFFFF));
      expect(parseHexColor('#123456'), equals(0xFF123456));
      // #RRGGBBAA -> AARRGGBB
      expect(parseHexColor('#12345680'), equals(0x80123456));
    });

    test(
      'ColorTheme.fromJson parses custom JSON themes with colors and tokenColors',
      () {
        const jsonStr = '''
      {
        "name": "Custom Theme",
        "colors": {
          "editor.foreground": "#CCCCCC",
          "editor.background": "#111111"
        },
        "tokenColors": [
          {
            "scope": "comment, punctuation.definition.comment",
            "settings": {
              "foreground": "#00FF00",
              "fontStyle": "italic bold"
            }
          },
          {
            "scope": ["string", "constant.other"],
            "settings": {
              "foreground": "#FF00FF"
            }
          }
        ]
      }
      ''';

        final theme = ColorTheme.fromJsonString(jsonStr);
        expect(theme.name, equals('Custom Theme'));
        expect(theme.defaultStyle.foreground, equals(0xFFCCCCCC));
        expect(theme.defaultStyle.background, equals(0xFF111111));

        final commentStyle = theme.resolveStyle([StyleScope('comment.line')]);
        expect(commentStyle.foreground, equals(0xFF00FF00));
        expect(commentStyle.italic, isTrue);
        expect(commentStyle.bold, isTrue);

        final stringStyle = theme.resolveStyle([StyleScope('string.quoted')]);
        expect(stringStyle.foreground, equals(0xFFFF00FF));
      },
    );
  });

  group('ScopeMatcher compound scope selectors & specificity', () {
    test('Single scope matching and dot prefix specificity', () {
      final scopes = [StyleScope('entity.name.function.dart')];

      final exactScore = ScopeMatcher.matchScore(
        scopes,
        'entity.name.function.dart',
      );
      final prefixScore = ScopeMatcher.matchScore(
        scopes,
        'entity.name.function',
      );
      final generalScore = ScopeMatcher.matchScore(scopes, 'entity');

      expect(exactScore, greaterThan(prefixScore));
      expect(prefixScore, greaterThan(generalScore));
      expect(generalScore, greaterThan(0));
      expect(ScopeMatcher.matchScore(scopes, 'keyword'), equals(0));
    });

    test('Compound scope selector matching ancestor scope stack', () {
      final scopeStack = [
        StyleScope('source.dart'),
        StyleScope('meta.function.dart'),
        StyleScope('entity.name.function.dart'),
      ];

      final compoundScore = ScopeMatcher.matchScore(
        scopeStack,
        'source.dart entity.name.function',
      );
      expect(compoundScore, greaterThan(0));

      final simpleScore = ScopeMatcher.matchScore(
        scopeStack,
        'entity.name.function',
      );
      expect(compoundScore, greaterThan(simpleScore));

      // Non-matching ancestor scope in compound selector must return 0
      final mismatchedCompound = ScopeMatcher.matchScore(
        scopeStack,
        'source.java entity.name.function',
      );
      expect(mismatchedCompound, equals(0));
    });

    test('Comma-separated selector list matching', () {
      final scopes = [StyleScope('string.quoted.double.dart')];
      final score = ScopeMatcher.matchScore(
        scopes,
        'comment, string.quoted, keyword',
      );
      expect(score, greaterThan(0));
    });
  });
}
