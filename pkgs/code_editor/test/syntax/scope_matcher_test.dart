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
  group('ScopeMatcher & ColorTheme', () {
    test('matchScore computes specificity correctly', () {
      final scopes = [
        StyleScope('source.dart'),
        StyleScope('meta.declaration.dart'),
        StyleScope('string.quoted.double.dart'),
      ];

      final score1 = ScopeMatcher.matchScore(scopes, 'string');
      final score2 = ScopeMatcher.matchScore(scopes, 'string.quoted');
      final score3 = ScopeMatcher.matchScore(scopes, 'string.quoted.double');
      final score4 = ScopeMatcher.matchScore(scopes, 'string.unquoted');

      expect(score1, greaterThan(0));
      expect(score2, greaterThan(score1));
      expect(score3, greaterThan(score2));
      expect(score4, equals(0));

      final scoreAncestor = ScopeMatcher.matchScore(
        scopes,
        'source.dart string.quoted.double',
      );
      expect(scoreAncestor, greaterThan(score3));

      final scoreMismatchAncestor = ScopeMatcher.matchScore(
        scopes,
        'source.python string.quoted.double',
      );
      expect(scoreMismatchAncestor, equals(0));
    });

    test('parseHexColor converts RGB, RRGGBB, RRGGBBAA to ARGB int', () {
      expect(parseHexColor('#FFF'), equals(0xFFFFFFFF));
      expect(parseHexColor('#FF0000'), equals(0xFFFF0000));
      expect(parseHexColor('#00FF0080'), equals(0x8000FF00));
    });

    test('ColorTheme resolves style rules based on scope specificity', () {
      final themeJson = {
        'name': 'Test Theme',
        'tokenColors': [
          {
            'scope': 'comment',
            'settings': {'foreground': '#00FF00', 'fontStyle': 'italic'},
          },
          {
            'scope': 'comment.block',
            'settings': {'foreground': '#008800', 'fontStyle': 'bold'},
          },
          {
            'scope': 'keyword',
            'settings': {'foreground': '#FF0000'},
          },
        ],
      };

      final theme = ColorTheme.fromJson(themeJson);

      final commentScopes = [StyleScope('comment.block.dart')];
      final style = theme.resolveStyle(commentScopes);

      expect(style.foreground, equals(0xFF008800));
      expect(style.bold, isTrue);

      final styleCache = StyleCache(theme);
      final cachedStyle = styleCache.getStyle(commentScopes);
      expect(cachedStyle, equals(style));
    });
  });
}
