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

/// Engine for cleanly formatting Dart code with customizable indentation and spacing rules.
@immutable
final class DartFormatterEngine {
  final int tabSize;

  const DartFormatterEngine({this.tabSize = 2});

  /// Formats Dart code [source] with consistent indentation, brace placement,
  /// operator spacing, and argument wrapping.
  String formatCode(String source) {
    if (source.trim().isEmpty) return '';

    final lines = source.split(RegExp(r'\r?\n'));
    final formattedLines = <String>[];
    int indentLevel = 0;
    bool inMultilineComment = false;
    bool inMultilineString = false;

    for (int i = 0; i < lines.length; i++) {
      final String rawLine = lines[i];
      final trimmed = rawLine.trim();

      if (trimmed.isEmpty) {
        // Only emit blank line if not consecutive blank lines
        if (formattedLines.isNotEmpty && formattedLines.last.isNotEmpty) {
          formattedLines.add('');
        }
        continue;
      }

      // Check continuation of multiline comment
      if (inMultilineComment) {
        if (trimmed.endsWith('*/') || trimmed.contains('*/')) {
          inMultilineComment = false;
        }
        formattedLines.add('${' ' * (indentLevel * tabSize)}$trimmed');
        continue;
      }

      // Check multiline strings
      if (inMultilineString) {
        if (trimmed.endsWith("'''") || trimmed.endsWith('"""')) {
          inMultilineString = false;
        }
        formattedLines.add(rawLine); // Preserve verbatim multiline string
        continue;
      }

      if (trimmed.startsWith('/*')) {
        inMultilineComment = !trimmed.contains('*/');
        formattedLines.add('${' ' * (indentLevel * tabSize)}$trimmed');
        continue;
      }

      final normalized = _normalizeLineSpacing(trimmed);
      final subLines = normalized.split('\n');

      for (final sub in subLines) {
        final subTrimmed = sub.trim();
        if (subTrimmed.isEmpty) continue;

        int lineOutdent = 0;
        if (subTrimmed.startsWith('}') ||
            subTrimmed.startsWith(']') ||
            subTrimmed.startsWith(')')) {
          lineOutdent = 1;
        }

        final effectiveIndent = (indentLevel - lineOutdent).clamp(0, 999);
        final indentStr = ' ' * (effectiveIndent * tabSize);
        formattedLines.add('$indentStr$subTrimmed');

        final netDelta = _calculateIndentDelta(subTrimmed);
        indentLevel = (indentLevel + netDelta).clamp(0, 999);
      }

      // Check if line opens a multiline string ''' or """
      if ((normalized.contains("'''") &&
              _countSubstrings(normalized, "'''") % 2 != 0) ||
          (normalized.contains('"""') &&
              _countSubstrings(normalized, '"""') % 2 != 0)) {
        inMultilineString = true;
      }
    }

    // Remove trailing blank lines
    while (formattedLines.isNotEmpty && formattedLines.last.isEmpty) {
      formattedLines.removeLast();
    }

    return formattedLines.join('\n');
  }

  int _countSubstrings(String str, String sub) {
    int count = 0;
    int pos = 0;
    while ((pos = str.indexOf(sub, pos)) != -1) {
      count++;
      pos += sub.length;
    }
    return count;
  }

  String _normalizeLineSpacing(String line) {
    // Preserve string literals or single-line comments verbatim
    if (line.startsWith('//') || line.startsWith('///')) {
      return line;
    }

    // Space after commas (if not already spaced or inside string)
    String res = line.replaceAllMapped(RegExp(r',([^\s])'), (m) => ', ${m[1]}');

    // Space after semicolons in for-loop headers
    res = res.replaceAllMapped(RegExp(r';([^\s])'), (m) => '; ${m[1]}');

    // Space around comparison and compound assignment operators first
    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*==\s*(\S)'),
      (m) => '${m[1]} == ${m[2]}',
    );
    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*!=\s*(\S)'),
      (m) => '${m[1]} != ${m[2]}',
    );
    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*<=\s*(\S)'),
      (m) => '${m[1]} <= ${m[2]}',
    );
    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*>=\s*(\S)'),
      (m) => '${m[1]} >= ${m[2]}',
    );

    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*\+=\s*(\S)'),
      (m) => '${m[1]} += ${m[2]}',
    );
    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*-=\s*(\S)'),
      (m) => '${m[1]} -= ${m[2]}',
    );
    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*\*=\s*(\S)'),
      (m) => '${m[1]} *= ${m[2]}',
    );
    res = res.replaceAllMapped(
      RegExp(r'(\S)\s*/=\s*(\S)'),
      (m) => '${m[1]} /= ${m[2]}',
    );

    // Space around plain assignment operator = (when not preceded by comparison/compound char)
    res = res.replaceAllMapped(
      RegExp(r'([^\s=!<>+\-*/])\s*=\s*(\S)'),
      (m) => '${m[1]} = ${m[2]}',
    );

    // Space around comparison < (never space generic type parameters like <Float64>)
    res = res.replaceAllMapped(
      RegExp(
        r'(\S)\s*<\s*(?!(Float64|Float32|Int32|Int64|Complex64|Complex128|int|double|bool|num|void|String|List|Map|Set|Future|[A-Z])\b)(\S)',
      ),
      (m) => '${m[1]} < ${m[3]}',
    );

    // Space around comparison > (never space generic closing > followed by ( ) , ; >)
    res = res.replaceAllMapped(
      RegExp(r'([a-zA-Z0-9_)\]])\s*>\s*([a-zA-Z0-9_\-])'),
      (m) => '${m[1]} > ${m[2]}',
    );

    // Space around control flow keywords
    res = res.replaceAllMapped(
      RegExp(r'\b(if|for|while|switch|catch)\('),
      (m) => '${m[1]} (',
    );

    // Ensure space before opening brace { when preceded by closing paren or identifier
    res = res.replaceAllMapped(
      RegExp(r'(\)|[a-zA-Z0-9_])\{'),
      (m) => '${m[1]} {',
    );

    // Split after opening brace { when followed by statement on the same line
    res = res.replaceAllMapped(RegExp(r'\{\s*([^\s}/*])'), (m) => '{\n${m[1]}');

    // Split before closing brace } when preceded by statement on the same line
    res = res.replaceAllMapped(RegExp(r'([^\s{/*])\s*\}'), (m) => '${m[1]}\n}');

    // Split multiple statements separated by semicolon onto new lines if not in for loop
    if (!res.startsWith('for ') && !res.startsWith('for(')) {
      res = res.replaceAllMapped(
        RegExp(r';\s*([^\s}/*])'),
        (m) => ';\n${m[1]}',
      );
    }

    // Trim double spaces not in string literals
    return res.trim();
  }

  int _calculateIndentDelta(String line) {
    // Ignore trailing single-line comments when computing depth
    final commentIdx = line.indexOf('//');
    final codePart = commentIdx != -1 ? line.substring(0, commentIdx) : line;

    final int openBraces =
        _countChar(codePart, '{') +
        _countChar(codePart, '[') +
        _countChar(codePart, '(');
    final int closeBraces =
        _countChar(codePart, '}') +
        _countChar(codePart, ']') +
        _countChar(codePart, ')');
    return openBraces - closeBraces;
  }

  int _countChar(String str, String ch) {
    int count = 0;
    for (int i = 0; i < str.length; i++) {
      if (str[i] == ch) count++;
    }
    return count;
  }
}
