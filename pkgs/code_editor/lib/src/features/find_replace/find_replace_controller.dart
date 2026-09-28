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

import '../../core/buffer/text_buffer.dart';
import '../../core/selection/selection.dart';
import '../../core/selection/text_position.dart';
import 'search_match.dart';

/// Controller managing text search and replace operations over a text buffer.
final class FindReplaceController {
  SearchOptions _options = const SearchOptions(query: '');
  List<SearchMatch> _matches = const [];
  int _activeMatchIndex = -1;

  /// Current search options.
  SearchOptions get options => _options;

  /// All active search matches in the document.
  List<SearchMatch> get matches => List.unmodifiable(_matches);

  /// Total count of matching occurrences.
  int get matchCount => _matches.length;

  /// Currently active match index (0-based, or -1 if no matches).
  int get activeMatchIndex => _activeMatchIndex;

  /// Gets the currently active [SearchMatch], or `null` if none.
  SearchMatch? get activeMatch =>
      (_activeMatchIndex >= 0 && _activeMatchIndex < _matches.length)
      ? _matches[_activeMatchIndex]
      : null;

  /// Executes a search across [buffer] with the given [options].
  void find(TextBuffer buffer, SearchOptions options) {
    _options = options;
    _matches = _computeMatches(buffer, options);

    if (_matches.isEmpty) {
      _activeMatchIndex = -1;
    } else if (_activeMatchIndex < 0 || _activeMatchIndex >= _matches.length) {
      _activeMatchIndex = 0;
    }
  }

  /// Advances to the next match forward in the document.
  SearchMatch? findNext() {
    if (_matches.isEmpty) return null;
    _activeMatchIndex = (_activeMatchIndex + 1) % _matches.length;
    return activeMatch;
  }

  /// Steps backward to the previous match in the document.
  SearchMatch? findPrevious() {
    if (_matches.isEmpty) return null;
    _activeMatchIndex =
        (_activeMatchIndex - 1 + _matches.length) % _matches.length;
    return activeMatch;
  }

  /// Clears active search state and matches.
  void clear() {
    _options = const SearchOptions(query: '');
    _matches = const [];
    _activeMatchIndex = -1;
  }

  List<SearchMatch> _computeMatches(TextBuffer buffer, SearchOptions opts) {
    if (opts.query.isEmpty || buffer.length == 0) return const [];

    final text = buffer.text;
    final results = <SearchMatch>[];

    RegExp regex;
    try {
      String pattern = opts.query;
      if (!opts.isRegex) {
        pattern = RegExp.escape(pattern);
      }
      if (opts.matchWholeWord) {
        pattern = r'\b' + pattern + r'\b';
      }
      regex = RegExp(pattern, caseSensitive: opts.matchCase);
    } catch (_) {
      return const [];
    }

    for (final m in regex.allMatches(text)) {
      final start = m.start;
      final end = m.end;
      final len = end - start;
      final matchedText = m.group(0) ?? '';

      final (startLine, startCol) = buffer.getLineAndColumnAt(start);
      final (endLine, endCol) = buffer.getLineAndColumnAt(end);

      results.add(
        SearchMatch(
          startOffset: start,
          length: len,
          range: TextSelection(
            base: TextPosition(startLine, startCol),
            extent: TextPosition(endLine, endCol),
          ),
          text: matchedText,
        ),
      );
    }

    return results;
  }
}
