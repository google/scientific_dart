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

import 'package:meta/meta.dart';

/// Basic token classifications.
enum TokenType {
  keyword,
  string,
  comment,
  number,
  operator,
  identifier,
  punctuation,
  whitespace,
  custom,
  unknown,
}

/// Represents a dot-separated hierarchical scope (e.g., `string.quoted.double.dart`).
@immutable
class StyleScope {
  final String raw;
  final List<String> segments;

  StyleScope(this.raw) : segments = raw.split('.');

  bool get isEmpty => raw.isEmpty;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is StyleScope &&
          runtimeType == other.runtimeType &&
          raw == other.raw;

  @override
  int get hashCode => raw.hashCode;

  @override
  String toString() => raw;
}

/// Resolved styling properties for a token.
@immutable
class ResolvedTokenStyle {
  /// Foreground color as ARGB int (e.g. 0xFF00FF00) or null.
  final int? foreground;

  /// Background color as ARGB int or null.
  final int? background;

  final bool bold;
  final bool italic;
  final bool underline;
  final bool strikethrough;

  const ResolvedTokenStyle({
    this.foreground,
    this.background,
    this.bold = false,
    this.italic = false,
    this.underline = false,
    this.strikethrough = false,
  });

  static const defaultStyle = ResolvedTokenStyle();

  ResolvedTokenStyle copyWith({
    int? foreground,
    int? background,
    bool? bold,
    bool? italic,
    bool? underline,
    bool? strikethrough,
  }) {
    return ResolvedTokenStyle(
      foreground: foreground ?? this.foreground,
      background: background ?? this.background,
      bold: bold ?? this.bold,
      italic: italic ?? this.italic,
      underline: underline ?? this.underline,
      strikethrough: strikethrough ?? this.strikethrough,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ResolvedTokenStyle &&
          runtimeType == other.runtimeType &&
          foreground == other.foreground &&
          background == other.background &&
          bold == other.bold &&
          italic == other.italic &&
          underline == other.underline &&
          strikethrough == other.strikethrough;

  @override
  int get hashCode => Object.hash(
    foreground,
    background,
    bold,
    italic,
    underline,
    strikethrough,
  );

  @override
  String toString() =>
      'ResolvedTokenStyle(fg: ${foreground?.toRadixString(16)}, bg: ${background?.toRadixString(16)}, bold: $bold, italic: $italic)';
}

/// Represents a tokenized range in a line.
@immutable
class SyntaxToken {
  final int offset;
  final int length;
  final TokenType type;
  final List<StyleScope> scopes;
  final ResolvedTokenStyle style;
  final String text;

  const SyntaxToken({
    required this.offset,
    required this.length,
    required this.type,
    this.scopes = const [],
    this.style = ResolvedTokenStyle.defaultStyle,
    this.text = '',
  });

  int get end => offset + length;

  SyntaxToken withStyle(ResolvedTokenStyle newStyle) {
    return SyntaxToken(
      offset: offset,
      length: length,
      type: type,
      scopes: scopes,
      style: newStyle,
      text: text,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SyntaxToken &&
          runtimeType == other.runtimeType &&
          offset == other.offset &&
          length == other.length &&
          type == other.type &&
          style == other.style &&
          text == other.text;

  @override
  int get hashCode => Object.hash(offset, length, type, style, text);

  @override
  String toString() =>
      'SyntaxToken(offset: $offset, len: $length, type: $type, scopes: $scopes, text: "$text")';
}
