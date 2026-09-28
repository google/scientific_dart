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
import '../../core/buffer/text_buffer.dart';
import '../../core/selection/selection.dart';
import '../../core/selection/text_position.dart';

/// Represents a defined code snippet template with placeholders ($1, $2, $0).
@immutable
final class CodeSnippet {
  /// Unique trigger prefix (e.g. `'for'`, `'if'`, `'array'`, `'fn'`).
  final String prefix;

  /// Display label or short description.
  final String label;

  /// Description of the snippet.
  final String description;

  /// The raw template string containing `$1`, `$2`, `$0`, `${1:default}`.
  final String body;

  /// Creates a code snippet definition.
  const CodeSnippet({
    required this.prefix,
    required this.label,
    required this.body,
    this.description = '',
  });

  /// Built-in rich Dart and NDArray code snippets.
  static const List<CodeSnippet> builtIn = [
    CodeSnippet(
      prefix: 'for',
      label: 'for loop',
      description: 'Standard for loop over indices',
      body: 'for (var \$1 = 0; \$1 < \$2; \$1++) {\n  \$0\n}',
    ),
    CodeSnippet(
      prefix: 'if',
      label: 'if statement',
      description: 'Conditional if block',
      body: 'if (\$1) {\n  \$0\n}',
    ),
    CodeSnippet(
      prefix: 'array',
      label: 'NDArray.create',
      description: 'Create an NDArray with shape and DType',
      body: 'final \$1 = NDArray.create([\$2], DType.\$3);\n\$0',
    ),
    CodeSnippet(
      prefix: 'fromList',
      label: 'NDArray.fromList',
      description: 'Create an NDArray from a Dart list',
      body: 'final \$1 = NDArray.fromList([\$2], [\$3], DType.\$4);\n\$0',
    ),
    CodeSnippet(
      prefix: 'fn',
      label: 'function declaration',
      description: 'Declare a Dart function',
      body: '\$1 \$2(\$3) {\n  \$0\n}',
    ),
    CodeSnippet(
      prefix: 'print',
      label: 'print statement',
      description: 'Print expression to console',
      body: 'print(\$1);\n\$0',
    ),
  ];
}

/// Represents an active tab stop position within an expanded snippet.
@immutable
final class TabStop {
  /// The numeric stop index (1, 2, ..., 0 is final stop).
  final int index;

  /// The character offset range in the document buffer.
  final int startOffset;
  final int length;

  /// Default placeholder text.
  final String placeholder;

  const TabStop({
    required this.index,
    required this.startOffset,
    required this.length,
    this.placeholder = '',
  });
}

/// Expanded snippet result with computed text and active tab stops.
@immutable
final class ExpandedSnippetResult {
  /// The formatted text to insert.
  final String insertedText;

  /// Ordered tab stops in the inserted text.
  final List<TabStop> tabStops;

  /// Initial selection for the first tab stop (or `$0` if none).
  final TextSelection initialSelection;

  const ExpandedSnippetResult({
    required this.insertedText,
    required this.tabStops,
    required this.initialSelection,
  });
}

/// Engine for expanding code snippets with tab stops and default values.
final class SnippetEngine {
  final Map<String, CodeSnippet> _snippets = {};

  SnippetEngine({List<CodeSnippet> snippets = CodeSnippet.builtIn}) {
    for (final s in snippets) {
      _snippets[s.prefix] = s;
    }
  }

  /// Finds a matching snippet for [prefix].
  CodeSnippet? findSnippet(String prefix) => _snippets[prefix];

  /// List of all registered snippets.
  List<CodeSnippet> get allSnippets => _snippets.values.toList();

  /// Expands a snippet [body] at [insertOffset] with [baseIndent].
  ExpandedSnippetResult expandSnippet({
    required String body,
    required int insertOffset,
    required TextBuffer buffer,
  }) {
    final (startLine, _) = buffer.getLineAndColumnAt(insertOffset);
    final lineOffset = buffer.getLineOffset(startLine);
    final lineLen = buffer.getLineLength(startLine);
    final lineText = buffer.getTextInRange(lineOffset, lineLen);

    // Compute base indentation
    int indentSpaces = 0;
    while (indentSpaces < lineText.length &&
        (lineText[indentSpaces] == ' ' || lineText[indentSpaces] == '\t')) {
      indentSpaces++;
    }
    final indentStr = lineText.substring(0, indentSpaces);

    // Regex for placeholders: ${1:default} or $1
    final regex = RegExp(r'\$\{(\d+):([^}]+)\}|\$(\d+)');
    final tabStopsMap = <int, List<TabStop>>{};
    final sb = StringBuffer();

    int cursor = 0;
    int currentInsertedOffset = insertOffset;

    for (final m in regex.allMatches(body)) {
      if (m.start > cursor) {
        final literal = body.substring(cursor, m.start);
        final indentedLiteral = literal.replaceAll('\n', '\n$indentStr');
        sb.write(indentedLiteral);
        currentInsertedOffset += indentedLiteral.length;
      }

      int index = 0;
      String defaultText = '';

      if (m.group(1) != null) {
        index = int.parse(m.group(1)!);
        defaultText = m.group(2)!;
      } else if (m.group(3) != null) {
        index = int.parse(m.group(3)!);
      }

      sb.write(defaultText);
      final stop = TabStop(
        index: index,
        startOffset: currentInsertedOffset,
        length: defaultText.length,
        placeholder: defaultText,
      );

      tabStopsMap.putIfAbsent(index, () => []).add(stop);
      currentInsertedOffset += defaultText.length;
      cursor = m.end;
    }

    if (cursor < body.length) {
      final literal = body.substring(cursor);
      final indentedLiteral = literal.replaceAll('\n', '\n$indentStr');
      sb.write(indentedLiteral);
    }

    final insertedString = sb.toString();
    final orderedStops = <TabStop>[];

    // Sort 1..N first, then 0 (final stop)
    final keys = tabStopsMap.keys.where((k) => k != 0).toList()..sort();
    for (final k in keys) {
      orderedStops.addAll(tabStopsMap[k]!);
    }
    if (tabStopsMap.containsKey(0)) {
      orderedStops.addAll(tabStopsMap[0]!);
    }

    TextSelection initialSel;
    if (orderedStops.isNotEmpty) {
      final firstStop = orderedStops.first;
      final (startL, startC) = buffer.getLineAndColumnAt(firstStop.startOffset);
      final (endL, endC) = buffer.getLineAndColumnAt(
        firstStop.startOffset + firstStop.length,
      );
      initialSel = TextSelection(
        base: TextPosition(startL, startC),
        extent: TextPosition(endL, endC),
      );
    } else {
      final (endL, endC) = buffer.getLineAndColumnAt(
        insertOffset + insertedString.length,
      );
      initialSel = TextSelection.collapsed(TextPosition(endL, endC));
    }

    return ExpandedSnippetResult(
      insertedText: insertedString,
      tabStops: orderedStops,
      initialSelection: initialSel,
    );
  }
}
