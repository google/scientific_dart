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
import 'package:meta/meta.dart';
import 'scope_matcher.dart';
import 'syntax_token.dart';

/// A single TextMate theme rule mapping scope selectors to token style overrides.
@immutable
final class ThemeRule {
  /// List of TextMate scope selectors for this rule.
  final List<String> scopes;

  /// Foreground ARGB color value.
  final int? foreground;

  /// Background ARGB color value.
  final int? background;

  /// Font style bold flag override.
  final bool? bold;

  /// Font style italic flag override.
  final bool? italic;

  /// Font style underline flag override.
  final bool? underline;

  /// Font style strikethrough flag override.
  final bool? strikethrough;

  const ThemeRule({
    required this.scopes,
    this.foreground,
    this.background,
    this.bold,
    this.italic,
    this.underline,
    this.strikethrough,
  });

  /// Constructs a [ThemeRule] from a VS Code / TextMate theme JSON rule object.
  factory ThemeRule.fromJson(Map<String, dynamic> json) {
    List<String> scopesList = [];
    final scopeVal = json['scope'];
    if (scopeVal is String) {
      scopesList = scopeVal
          .split(',')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();
    } else if (scopeVal is List) {
      for (final e in scopeVal) {
        if (e is String) {
          scopesList.addAll(
            e.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty),
          );
        }
      }
    }

    final settings = json['settings'] as Map<String, dynamic>? ?? {};
    final fgStr = settings['foreground'] as String?;
    final bgStr = settings['background'] as String?;
    final fontStyle = settings['fontStyle'] as String? ?? '';

    final isBold = fontStyle.contains('bold');
    final isItalic = fontStyle.contains('italic');
    final isUnderline = fontStyle.contains('underline');
    final isStrikethrough = fontStyle.contains('strikethrough');

    return ThemeRule(
      scopes: scopesList,
      foreground: fgStr != null ? parseHexColor(fgStr) : null,
      background: bgStr != null ? parseHexColor(bgStr) : null,
      bold: isBold ? true : null,
      italic: isItalic ? true : null,
      underline: isUnderline ? true : null,
      strikethrough: isStrikethrough ? true : null,
    );
  }
}

/// Helper utility to parse Hex color strings (`#RGB`, `#RRGGBB`, or `#RRGGBBAA`) to ARGB integer.
int? parseHexColor(String hex) {
  var clean = hex.trim().replaceAll('#', '');
  if (clean.isEmpty) return null;
  if (clean.length == 3) {
    final r = clean[0];
    final g = clean[1];
    final b = clean[2];
    clean = '$r$r$g$g$b$b';
  }
  if (clean.length == 6) {
    clean = 'FF$clean';
  } else if (clean.length == 8) {
    final rrggbb = clean.substring(0, 6);
    final aa = clean.substring(6, 8);
    clean = '$aa$rrggbb';
  }
  return int.tryParse(clean, radix: 16);
}

/// Complete TextMate / VS Code color theme model with scope style resolution.
@immutable
final class ColorTheme {
  /// Name of the theme (e.g. "Dark+", "Monokai").
  final String name;

  /// Ordered list of theme rules.
  final List<ThemeRule> rules;

  /// Default token style when no rules match.
  final ResolvedTokenStyle defaultStyle;

  const ColorTheme({
    required this.name,
    required this.rules,
    this.defaultStyle = const ResolvedTokenStyle(
      foreground: 0xFFD4D4D4,
      background: 0xFF1E1E1E,
    ),
  });

  /// Constructs a [ColorTheme] from a VS Code theme JSON object.
  factory ColorTheme.fromJson(Map<String, dynamic> json) {
    final name = json['name'] as String? ?? 'Unnamed Theme';

    // Parse editor background/foreground colors from json['colors'] map if present
    final colors = json['colors'] as Map<String, dynamic>?;
    int fgColor = 0xFFD4D4D4;
    int bgColor = 0xFF1E1E1E;

    if (colors != null) {
      final fgStr =
          (colors['editor.foreground'] ?? colors['foreground']) as String?;
      final bgStr =
          (colors['editor.background'] ?? colors['background']) as String?;
      if (fgStr != null) {
        final parsed = parseHexColor(fgStr);
        if (parsed != null) fgColor = parsed;
      }
      if (bgStr != null) {
        final parsed = parseHexColor(bgStr);
        if (parsed != null) bgColor = parsed;
      }
    }

    final tokenColors = json['tokenColors'] as List? ?? [];
    final rules = <ThemeRule>[];
    for (final item in tokenColors) {
      if (item is Map<String, dynamic>) {
        rules.add(ThemeRule.fromJson(item));
      }
    }

    return ColorTheme(
      name: name,
      rules: rules,
      defaultStyle: ResolvedTokenStyle(
        foreground: fgColor,
        background: bgColor,
      ),
    );
  }

  /// Parses a VS Code JSON theme string into a [ColorTheme].
  factory ColorTheme.fromJsonString(String jsonStr) {
    return ColorTheme.fromJson(jsonDecode(jsonStr) as Map<String, dynamic>);
  }

  /// Built-in VS Code Dark+ standard theme.
  factory ColorTheme.darkPlus() {
    return const ColorTheme(
      name: 'Dark+',
      defaultStyle: ResolvedTokenStyle(
        foreground: 0xFFD4D4D4,
        background: 0xFF1E1E1E,
      ),
      rules: [
        ThemeRule(
          scopes: ['comment', 'punctuation.definition.comment'],
          foreground: 0xFF6A9955,
          italic: true,
        ),
        ThemeRule(
          scopes: ['string', 'string.quoted', 'string.template'],
          foreground: 0xFFCE9178,
        ),
        ThemeRule(
          scopes: [
            'keyword',
            'keyword.control',
            'storage.type',
            'storage.modifier',
          ],
          foreground: 0xFF569CD6,
        ),
        ThemeRule(
          scopes: ['entity.name.function', 'support.function'],
          foreground: 0xFFDCDCAA,
        ),
        ThemeRule(
          scopes: ['constant.numeric', 'number'],
          foreground: 0xFFB5CEA8,
        ),
        ThemeRule(
          scopes: [
            'entity.name.type',
            'entity.name.class',
            'support.class',
            'support.type',
            'storage.type.primitive',
          ],
          foreground: 0xFF4EC9B0,
        ),
        ThemeRule(
          scopes: ['variable', 'variable.other', 'variable.parameter'],
          foreground: 0xFF9CDCFE,
        ),
        ThemeRule(
          scopes: ['constant', 'constant.language', 'constant.character'],
          foreground: 0xFF4FC1FF,
        ),
        ThemeRule(
          scopes: ['constant.character.escape'],
          foreground: 0xFFD7BA7D,
        ),
        ThemeRule(
          scopes: [
            'punctuation.section.embedded',
            'variable.other.interpolation',
          ],
          foreground: 0xFF569CD6,
        ),
        ThemeRule(
          scopes: ['storage.type.annotation', 'meta.declaration.annotation'],
          foreground: 0xFFDCDCAA,
        ),
      ],
    );
  }

  /// Built-in Catppuccin Mocha theme matching the Scientific Dart Notebook UI.
  factory ColorTheme.catppuccinMocha() {
    return const ColorTheme(
      name: 'Catppuccin Mocha',
      defaultStyle: ResolvedTokenStyle(
        foreground: 0xFFCDD6F4,
        background: 0xFF1E1E2E,
      ),
      rules: [
        ThemeRule(
          scopes: ['comment', 'punctuation.definition.comment'],
          foreground: 0xFF6C7086,
          italic: true,
        ),
        ThemeRule(
          scopes: [
            'comment.line.documentation',
            'comment.block.documentation',
            'comment.line.doc',
          ],
          foreground: 0xFF7F849C,
          italic: true,
        ),
        ThemeRule(
          scopes: ['markup.inline.raw'],
          foreground: 0xFFA6E3A1,
          italic: false,
        ),
        ThemeRule(
          scopes: ['variable.other.link'],
          foreground: 0xFF89B4FA,
          italic: false,
        ),
        ThemeRule(
          scopes: ['string', 'string.quoted', 'string.template'],
          foreground: 0xFFA6E3A1,
        ),
        ThemeRule(
          scopes: ['constant.character.escape'],
          foreground: 0xFFF5C2E7,
        ),
        ThemeRule(
          scopes: [
            'punctuation.section.embedded',
            'variable.other.interpolation',
          ],
          foreground: 0xFFF38BA8,
        ),
        ThemeRule(
          scopes: ['keyword', 'keyword.control'],
          foreground: 0xFFCBA6F7,
          bold: true,
        ),
        ThemeRule(
          scopes: ['keyword.declaration', 'storage.type', 'storage.modifier'],
          foreground: 0xFFCBA6F7,
        ),
        ThemeRule(
          scopes: ['storage.type.annotation', 'meta.declaration.annotation'],
          foreground: 0xFFF9E2AF,
        ),
        ThemeRule(
          scopes: ['constant.language'],
          foreground: 0xFFFAB387,
          bold: true,
        ),
        ThemeRule(
          scopes: ['variable.language'],
          foreground: 0xFFF38BA8,
          italic: true,
        ),
        ThemeRule(
          scopes: ['entity.name.function', 'support.function'],
          foreground: 0xFF89B4FA,
        ),
        ThemeRule(
          scopes: ['constant.numeric', 'number'],
          foreground: 0xFFFAB387,
        ),
        ThemeRule(
          scopes: [
            'entity.name.type',
            'entity.name.class',
            'support.class',
            'support.type',
            'storage.type.primitive',
          ],
          foreground: 0xFFF9E2AF,
        ),
        ThemeRule(
          scopes: ['variable.parameter.named'],
          foreground: 0xFFEBA0AC,
          italic: true,
        ),
        ThemeRule(scopes: ['variable.other.property'], foreground: 0xFF94E2D5),
        ThemeRule(
          scopes: ['variable', 'variable.other', 'variable.parameter'],
          foreground: 0xFFCDD6F4,
        ),
        ThemeRule(scopes: ['keyword.operator'], foreground: 0xFF89DCEB),
        ThemeRule(
          scopes: [
            'punctuation',
            'punctuation.terminator',
            'punctuation.section',
          ],
          foreground: 0xFF9399B2,
        ),
      ],
    );
  }

  /// Built-in VS Code Light+ standard theme.
  factory ColorTheme.lightPlus() {
    return const ColorTheme(
      name: 'Light+',
      defaultStyle: ResolvedTokenStyle(
        foreground: 0xFF000000,
        background: 0xFFFFFFFF,
      ),
      rules: [
        ThemeRule(
          scopes: ['comment', 'punctuation.definition.comment'],
          foreground: 0xFF008000,
          italic: true,
        ),
        ThemeRule(scopes: ['string', 'string.quoted'], foreground: 0xFFA31515),
        ThemeRule(
          scopes: [
            'keyword',
            'keyword.control',
            'storage.type',
            'storage.modifier',
          ],
          foreground: 0xFF0000FF,
        ),
        ThemeRule(
          scopes: ['entity.name.function', 'support.function'],
          foreground: 0xFF795E26,
        ),
        ThemeRule(
          scopes: ['constant.numeric', 'number'],
          foreground: 0xFF098658,
        ),
        ThemeRule(
          scopes: [
            'entity.name.type',
            'entity.name.class',
            'support.class',
            'storage.type.primitive',
          ],
          foreground: 0xFF267F99,
        ),
        ThemeRule(
          scopes: ['variable', 'variable.other'],
          foreground: 0xFF001080,
        ),
        ThemeRule(
          scopes: ['constant', 'constant.language'],
          foreground: 0xFF0070C1,
        ),
      ],
    );
  }

  /// Built-in Monokai standard theme.
  factory ColorTheme.monokai() {
    return const ColorTheme(
      name: 'Monokai',
      defaultStyle: ResolvedTokenStyle(
        foreground: 0xFFF8F8F2,
        background: 0xFF272822,
      ),
      rules: [
        ThemeRule(
          scopes: ['comment', 'punctuation.definition.comment'],
          foreground: 0xFF75715E,
          italic: true,
        ),
        ThemeRule(scopes: ['string', 'string.quoted'], foreground: 0xFFE6DB74),
        ThemeRule(
          scopes: ['keyword', 'keyword.control', 'storage.type'],
          foreground: 0xFFF92672,
        ),
        ThemeRule(
          scopes: ['entity.name.function', 'support.function'],
          foreground: 0xFFA6E22E,
        ),
        ThemeRule(
          scopes: ['constant.numeric', 'number'],
          foreground: 0xFFAE81FF,
        ),
        ThemeRule(
          scopes: ['entity.name.type', 'entity.name.class'],
          foreground: 0xFF66D9EF,
        ),
        ThemeRule(
          scopes: ['variable', 'variable.other', 'variable.parameter'],
          foreground: 0xFFFD971F,
        ),
      ],
    );
  }

  /// Built-in Solarized Dark standard theme.
  factory ColorTheme.solarizedDark() {
    return const ColorTheme(
      name: 'Solarized Dark',
      defaultStyle: ResolvedTokenStyle(
        foreground: 0xFF839496,
        background: 0xFF002B36,
      ),
      rules: [
        ThemeRule(
          scopes: ['comment', 'punctuation.definition.comment'],
          foreground: 0xFF586E75,
          italic: true,
        ),
        ThemeRule(scopes: ['string', 'string.quoted'], foreground: 0xFF2AA198),
        ThemeRule(
          scopes: ['keyword', 'keyword.control', 'storage.type'],
          foreground: 0xFF859900,
        ),
        ThemeRule(
          scopes: ['entity.name.function', 'support.function'],
          foreground: 0xFF268BD2,
        ),
        ThemeRule(
          scopes: ['constant.numeric', 'number'],
          foreground: 0xFFD33682,
        ),
        ThemeRule(
          scopes: ['entity.name.type', 'entity.name.class'],
          foreground: 0xFFB58900,
        ),
        ThemeRule(
          scopes: ['variable', 'variable.other'],
          foreground: 0xFFCB4B16,
        ),
      ],
    );
  }

  /// Alias for [solarizedDark].
  factory ColorTheme.solarized() => ColorTheme.solarizedDark();

  /// Resolves effective token styling for a given list of hierarchical scope descriptors.
  ResolvedTokenStyle resolveStyle(List<StyleScope> scopes) {
    if (scopes.isEmpty) return defaultStyle;

    int? fg = defaultStyle.foreground;
    int? bg = defaultStyle.background;
    bool bold = defaultStyle.bold;
    bool italic = defaultStyle.italic;
    bool underline = defaultStyle.underline;
    bool strikethrough = defaultStyle.strikethrough;

    int bestFgScore = -1;
    int bestBgScore = -1;
    int bestBoldScore = -1;
    int bestItalicScore = -1;
    int bestUnderlineScore = -1;
    int bestStrikethroughScore = -1;

    for (final rule in rules) {
      for (final selector in rule.scopes) {
        final score = ScopeMatcher.matchScore(scopes, selector);
        if (score <= 0) continue;

        if (rule.foreground != null && score >= bestFgScore) {
          bestFgScore = score;
          fg = rule.foreground;
        }
        if (rule.background != null && score >= bestBgScore) {
          bestBgScore = score;
          bg = rule.background;
        }
        if (rule.bold != null && score >= bestBoldScore) {
          bestBoldScore = score;
          bold = rule.bold!;
        }
        if (rule.italic != null && score >= bestItalicScore) {
          bestItalicScore = score;
          italic = rule.italic!;
        }
        if (rule.underline != null && score >= bestUnderlineScore) {
          bestUnderlineScore = score;
          underline = rule.underline!;
        }
        if (rule.strikethrough != null && score >= bestStrikethroughScore) {
          bestStrikethroughScore = score;
          strikethrough = rule.strikethrough!;
        }
      }
    }

    return ResolvedTokenStyle(
      foreground: fg,
      background: bg,
      bold: bold,
      italic: italic,
      underline: underline,
      strikethrough: strikethrough,
    );
  }
}

/// Caches resolved styles for token scope hierarchies.
final class StyleCache {
  final ColorTheme theme;
  final Map<String, ResolvedTokenStyle> _cache = {};

  StyleCache(this.theme);

  ResolvedTokenStyle getStyle(List<StyleScope> scopes) {
    if (scopes.isEmpty) return theme.defaultStyle;
    final key = scopes.map((s) => s.raw).join('|');
    return _cache.putIfAbsent(key, () => theme.resolveStyle(scopes));
  }

  void clear() {
    _cache.clear();
  }
}
