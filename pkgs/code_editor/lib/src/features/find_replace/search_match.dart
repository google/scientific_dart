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
import '../../core/selection/selection.dart';

/// Represents a single search match in a text document.
@immutable
final class SearchMatch {
  /// The 0-based document absolute start character offset.
  final int startOffset;

  /// The length of the matched string.
  final int length;

  /// The line and column range of the match.
  final TextSelection range;

  /// The matched text content.
  final String text;

  /// Creates a [SearchMatch].
  const SearchMatch({
    required this.startOffset,
    required this.length,
    required this.range,
    required this.text,
  });

  /// The 0-based end offset.
  int get endOffset => startOffset + length;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SearchMatch &&
          runtimeType == other.runtimeType &&
          startOffset == other.startOffset &&
          length == other.length &&
          range == other.range &&
          text == other.text;

  @override
  int get hashCode => Object.hash(startOffset, length, range, text);

  @override
  String toString() => 'SearchMatch(offset: $startOffset..$endOffset, "$text")';
}

/// Search configuration query options.
@immutable
final class SearchOptions {
  /// The search query string or regex pattern.
  final String query;

  /// Whether the search is case-sensitive.
  final bool matchCase;

  /// Whether to match whole words only.
  final bool matchWholeWord;

  /// Whether [query] is a regular expression.
  final bool isRegex;

  /// Creates [SearchOptions].
  const SearchOptions({
    required this.query,
    this.matchCase = false,
    this.matchWholeWord = false,
    this.isRegex = false,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SearchOptions &&
          runtimeType == other.runtimeType &&
          query == other.query &&
          matchCase == other.matchCase &&
          matchWholeWord == other.matchWholeWord &&
          isRegex == other.isRegex;

  @override
  int get hashCode => Object.hash(query, matchCase, matchWholeWord, isRegex);
}
