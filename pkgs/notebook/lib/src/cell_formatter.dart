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

import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/token.dart';
import 'package:dart_style/dart_style.dart';

const String _asyncWrapperHeader = 'Future<void> __notebookCell__() async {';

/// Formats Dart notebook cell [code] using [DartFormatter].
///
/// Unlike a standalone `.dart` source file, a notebook cell may contain:
/// - Top-level declarations or directives (with or without a trailing `;`),
/// - One or more statements (`for`, `if`, `while`, local variables, `await`),
/// - A trailing expression without a semicolon (`a * 2`, `display(...)`), or
/// - A combination of `import` directives, top-level declarations, statements,
///   and a trailing expression.
///
/// If [code] contains syntax errors that prevent formatting, returns
/// `code.trim()` unchanged.
String formatNotebookCellCode(String code, {int pageWidth = 120}) {
  final trimmed = code.trim();
  if (trimmed.isEmpty) return trimmed;

  // Preserve notebook magic commands (`%pub add ...` / `pub add ...`).
  if (RegExp(
    r'^(?:%)?(?:pub\s+add|add)\s+([\w\d_\-]+)\s*;?$',
  ).hasMatch(trimmed)) {
    return trimmed;
  }

  // 1. Try formatting directly as a compilation unit (top-level declarations,
  //    directives, or comment-only cells).
  if (_tryFormatUnit(trimmed, pageWidth: pageWidth) case final formatted?) {
    return formatted;
  }

  // 2. Try formatting as a compilation unit after inserting a synthetic
  //    trailing semicolon (e.g. a single variable declaration without `;`).
  if (_insertSyntheticSemicolon(trimmed) case final withSemi?) {
    if (_tryFormatUnit(withSemi, pageWidth: pageWidth) case final formatted?) {
      return _removeSyntheticSemicolon(formatted);
    }
  }

  // 3. Try formatting as the body of an `async` function (optionally preceded
  //    by top-level directives such as `import`).
  if (_tryFormatAsAsyncCellBody(trimmed, pageWidth: pageWidth)
      case final formatted?) {
    return formatted;
  }

  // 4. Fallback for cells that mix top-level type declarations (`class`,
  //    `enum`, `mixin`, `extension`, `typedef`) with statements or expressions.
  if (_tryFormatMixedTopLevelAndStatements(trimmed, pageWidth: pageWidth)
      case final formatted?) {
    return formatted;
  }

  return trimmed;
}

String? _tryFormatUnit(String source, {required int pageWidth}) {
  try {
    final formatter = DartFormatter(
      languageVersion: DartFormatter.latestLanguageVersion,
      pageWidth: pageWidth,
    );
    return formatter.format(source).trim();
  } catch (_) {
    return null;
  }
}

String? _tryFormatAsAsyncCellBody(String source, {required int pageWidth}) {
  final (:directivesPreamble, :body) = _splitLeadingDirectives(source);
  if (body.isEmpty) {
    return directivesPreamble.isEmpty
        ? null
        : _tryFormatUnit(directivesPreamble, pageWidth: pageWidth);
  }

  // Try without adding a synthetic semicolon first.
  if (_tryFormatWrappedBody(
        directivesPreamble: directivesPreamble,
        body: body,
        addedTrailingSemicolon: false,
        pageWidth: pageWidth,
      )
      case final formatted?) {
    return formatted;
  }

  // Try with a synthetic semicolon appended after the final token in `body`.
  if (_insertSyntheticSemicolon(body) case final bodyWithSemi?) {
    if (_tryFormatWrappedBody(
          directivesPreamble: directivesPreamble,
          body: bodyWithSemi,
          addedTrailingSemicolon: true,
          pageWidth: pageWidth,
        )
        case final formatted?) {
      return formatted;
    }
  }

  return null;
}

String? _tryFormatWrappedBody({
  required String directivesPreamble,
  required String body,
  required bool addedTrailingSemicolon,
  required int pageWidth,
}) {
  final wrapper = StringBuffer();
  if (directivesPreamble.isNotEmpty) {
    wrapper.writeln(directivesPreamble);
    wrapper.writeln();
  }
  wrapper.writeln(_asyncWrapperHeader);
  wrapper.writeln(body);
  wrapper.writeln('}');

  final formattedWrapper = _tryFormatUnit(
    wrapper.toString(),
    pageWidth: pageWidth + 2,
  );
  if (formattedWrapper == null) return null;

  final markerIdx = formattedWrapper.indexOf(_asyncWrapperHeader);
  if (markerIdx == -1) return null;
  final closeBraceIdx = formattedWrapper.lastIndexOf('}');
  if (closeBraceIdx <= markerIdx) return null;

  final formattedPreamble = formattedWrapper.substring(0, markerIdx).trim();
  var innerBody = formattedWrapper.substring(
    markerIdx + _asyncWrapperHeader.length,
    closeBraceIdx,
  );
  if (innerBody.startsWith('\r\n')) {
    innerBody = innerBody.substring(2);
  } else if (innerBody.startsWith('\n')) {
    innerBody = innerBody.substring(1);
  }
  if (innerBody.endsWith('\r\n')) {
    innerBody = innerBody.substring(0, innerBody.length - 2);
  } else if (innerBody.endsWith('\n')) {
    innerBody = innerBody.substring(0, innerBody.length - 1);
  }

  var unindentedBody = _unindentBlockBody(innerBody).trim();
  if (addedTrailingSemicolon) {
    unindentedBody = _removeSyntheticSemicolon(unindentedBody);
  }

  if (formattedPreamble.isEmpty) {
    return unindentedBody;
  }
  if (unindentedBody.isEmpty) {
    return formattedPreamble;
  }
  return '$formattedPreamble\n\n$unindentedBody';
}

({String directivesPreamble, String body}) _splitLeadingDirectives(
  String source,
) {
  final parseResult = parseString(content: source, throwIfDiagnostics: false);
  final directives = parseResult.unit.directives;
  if (directives.isEmpty) {
    return (directivesPreamble: '', body: source);
  }
  final splitOffset = directives.last.end;
  return (
    directivesPreamble: source.substring(0, splitOffset).trim(),
    body: source.substring(splitOffset).trim(),
  );
}

Token? _lastNonSyntheticToken(String source) {
  final parseResult = parseString(content: source, throwIfDiagnostics: false);
  var token = parseResult.unit.beginToken;
  Token? lastReal;
  while (!token.isEof) {
    if (!token.isSynthetic) {
      lastReal = token;
    }
    final next = token.next;
    if (next == null || identical(next, token)) break;
    token = next;
  }
  return lastReal;
}

String? _insertSyntheticSemicolon(String source) {
  final endToken = _lastNonSyntheticToken(source);
  if (endToken == null ||
      endToken.type == TokenType.SEMICOLON ||
      endToken.type == TokenType.CLOSE_CURLY_BRACKET) {
    return null;
  }
  final insertOffset = endToken.end;
  if (insertOffset < 0 || insertOffset > source.length) return null;
  return '${source.substring(0, insertOffset)};${source.substring(insertOffset)}';
}

String _removeSyntheticSemicolon(String formatted) {
  final endToken = _lastNonSyntheticToken(formatted);
  if (endToken != null && endToken.type == TokenType.SEMICOLON) {
    return '${formatted.substring(0, endToken.offset)}${formatted.substring(endToken.end)}'
        .trimRight();
  }
  return formatted;
}

/// Removes one level (2 spaces) of indentation from each line of [indentedBody]
/// except for continuation lines inside multiline string literals.
String _unindentBlockBody(String indentedBody) {
  final parseResult = parseString(
    content: indentedBody,
    throwIfDiagnostics: false,
  );
  final multilineTokenSpans = <(int, int)>[];
  var token = parseResult.unit.beginToken;
  while (!token.isEof) {
    if (token.end > token.offset + 1) {
      final lexeme = token.lexeme;
      if (lexeme.contains('\n')) {
        multilineTokenSpans.add((token.offset, token.end));
      }
    }
    final next = token.next;
    if (next == null || identical(next, token)) break;
    token = next;
  }

  bool isOffsetInsideMultilineToken(int offset) {
    for (final (start, end) in multilineTokenSpans) {
      if (offset > start && offset < end) return true;
      if (start >= offset) break;
    }
    return false;
  }

  final sb = StringBuffer();
  var lineStart = 0;
  final len = indentedBody.length;

  while (lineStart <= len) {
    final nlIdx = indentedBody.indexOf('\n', lineStart);
    final lineEnd = nlIdx == -1 ? len : nlIdx;
    var line = indentedBody.substring(lineStart, lineEnd);

    if (!isOffsetInsideMultilineToken(lineStart)) {
      if (line.startsWith('  ')) {
        line = line.substring(2);
      } else if (line.startsWith(' ')) {
        line = line.substring(1);
      }
    }

    sb.write(line);
    if (nlIdx == -1) break;
    sb.write('\n');
    lineStart = nlIdx + 1;
  }

  return sb.toString();
}

String? _tryFormatMixedTopLevelAndStatements(
  String source, {
  required int pageWidth,
}) {
  final (:directivesPreamble, :body) = _splitLeadingDirectives(source);
  final chunks = _splitTopLevelChunks(body);
  if (chunks.length <= 1) return null;

  final formattedSections = <String>[];
  if (directivesPreamble.isNotEmpty) {
    final formattedDir = _tryFormatUnit(
      directivesPreamble,
      pageWidth: pageWidth,
    );
    if (formattedDir == null) return null;
    formattedSections.add(formattedDir);
  }

  final pendingStatements = <String>[];

  bool flushStatements() {
    if (pendingStatements.isEmpty) return true;
    final combined = pendingStatements.join('\n');
    pendingStatements.clear();
    final formattedStmts = _tryFormatAsAsyncCellBody(
      combined,
      pageWidth: pageWidth,
    );
    if (formattedStmts == null) return false;
    formattedSections.add(formattedStmts);
    return true;
  }

  for (final chunk in chunks) {
    if (_isTopLevelTypeDeclaration(chunk)) {
      if (!flushStatements()) return null;
      var formattedType = _tryFormatUnit(chunk, pageWidth: pageWidth);
      if (formattedType == null) {
        if (_insertSyntheticSemicolon(chunk) case final withSemi?) {
          if (_tryFormatUnit(withSemi, pageWidth: pageWidth)
              case final formatted?) {
            formattedType = _removeSyntheticSemicolon(formatted);
          }
        }
      }
      if (formattedType == null) return null;
      formattedSections.add(formattedType);
    } else {
      pendingStatements.add(chunk);
    }
  }

  if (!flushStatements()) return null;
  return formattedSections.join('\n\n');
}

bool _isTopLevelTypeDeclaration(String chunk) {
  final parseResult = parseString(content: chunk, throwIfDiagnostics: false);
  final firstToken = parseResult.unit.beginToken;
  if (firstToken.isEof) return false;
  final stripped = chunk.substring(firstToken.offset);
  return RegExp(
    r'^(?:abstract\s+|base\s+|final\s+|interface\s+|sealed\s+|mixin\s+)*(?:class|enum|mixin|extension|typedef)\b',
  ).hasMatch(stripped);
}

List<String> _splitTopLevelChunks(String source) {
  final chunks = <String>[];
  final parseResult = parseString(content: source, throwIfDiagnostics: false);
  var token = parseResult.unit.beginToken;
  var chunkStart = 0;

  while (!token.isEof) {
    final lexeme = token.lexeme;
    if (!token.isSynthetic &&
        (lexeme == '{' || lexeme == '(' || lexeme == '[')) {
      final endGroup = token.endGroup;
      if (endGroup != null && !endGroup.isSynthetic) {
        token = endGroup;
        if (lexeme == '{') {
          var next = token.next;
          while (next != null && !next.isEof && next.isSynthetic) {
            next = next.next;
          }
          final nextLexeme = (next == null || next.isEof) ? null : next.lexeme;
          final continuesBlock =
              nextLexeme == 'else' ||
              nextLexeme == 'catch' ||
              nextLexeme == 'on' ||
              nextLexeme == 'finally' ||
              nextLexeme == ';';
          if (!continuesBlock) {
            final endOffset = token.end;
            final slice = source.substring(chunkStart, endOffset).trim();
            if (slice.isNotEmpty) {
              chunks.add(slice);
            }
            chunkStart = endOffset;
          }
        }
      }
    } else if (!token.isSynthetic && token.type == TokenType.SEMICOLON) {
      final endOffset = token.end;
      final slice = source.substring(chunkStart, endOffset).trim();
      if (slice.isNotEmpty) {
        chunks.add(slice);
      }
      chunkStart = endOffset;
    }

    final next = token.next;
    if (next == null || identical(next, token)) break;
    token = next;
  }

  if (chunkStart < source.length) {
    final tail = source.substring(chunkStart).trim();
    if (tail.isNotEmpty) {
      chunks.add(tail);
    }
  }
  return chunks;
}
