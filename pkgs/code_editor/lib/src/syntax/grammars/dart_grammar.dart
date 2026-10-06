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

import '../syntax_token.dart';
import '../textmate_lexer.dart';

/// Comprehensive TextMate grammar rules for the Dart programming language (Dart 3.x).
final class DartGrammar {
  DartGrammar._();

  /// Creates a [TextMateLexer] configured with the complete Dart grammar.
  static TextMateLexer createLexer() {
    return TextMateLexer(rootRules: rules);
  }

  static List<TextMateRule> get _docCommentPatterns => [
    TextMateRule(
      id: 'dart.comment.doc.code',
      name: 'markup.inline.raw.string.markdown',
      type: TokenType.comment,
      match: RegExp(r'`[^`]+`'),
    ),
    TextMateRule(
      id: 'dart.comment.doc.reference',
      name: 'variable.other.link.dart',
      type: TokenType.comment,
      match: RegExp(r'\[[a-zA-Z_][\w.]*\]'),
    ),
  ];

  static List<TextMateRule> get _stringPatterns => [
    TextMateRule(
      id: 'dart.string.escape',
      name: 'constant.character.escape.dart',
      type: TokenType.string,
      match: RegExp(
        r'\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|u[0-9a-fA-F]{4}|.)',
      ),
    ),
    TextMateRule(
      id: 'dart.string.interpolation.expression',
      name: 'meta.embedded.expression.dart',
      beginName: 'punctuation.section.embedded.begin.dart',
      endName: 'punctuation.section.embedded.end.dart',
      type: TokenType.custom,
      beginType: TokenType.punctuation,
      endType: TokenType.punctuation,
      begin: RegExp(r'\$\{'),
      end: RegExp(r'\}'),
      includeRootRules: true,
      clearAncestorScopes: true,
      patterns: [
        TextMateRule(
          id: 'dart.interpolation.braces',
          name: 'meta.brace.curly.dart',
          beginName: 'punctuation.section.braces.begin.dart',
          endName: 'punctuation.section.braces.end.dart',
          type: TokenType.custom,
          beginType: TokenType.punctuation,
          endType: TokenType.punctuation,
          begin: RegExp(r'\{'),
          end: RegExp(r'\}'),
          includeRootRules: true,
        ),
      ],
    ),
    TextMateRule(
      id: 'dart.string.interpolation.variable',
      name: 'variable.other.interpolation.dart',
      type: TokenType.identifier,
      match: RegExp(r'\$[a-zA-Z_]\w*'),
    ),
  ];

  /// Complete list of root rules for Dart syntax highlighting.
  static List<TextMateRule> get rules => [
    // 1. Comments (Doc & Standard)
    TextMateRule(
      id: 'dart.comment.block.doc',
      name: 'comment.block.documentation.dart',
      type: TokenType.comment,
      begin: RegExp(r'/\*\*(?!\*)'),
      end: RegExp(r'\*/'),
      patterns: _docCommentPatterns,
    ),
    TextMateRule(
      id: 'dart.comment.block',
      name: 'comment.block.dart',
      type: TokenType.comment,
      begin: RegExp(r'/\*'),
      end: RegExp(r'\*/'),
    ),
    TextMateRule(
      id: 'dart.comment.doc',
      name: 'comment.line.documentation.dart',
      type: TokenType.comment,
      begin: RegExp(r'///'),
      end: RegExp(r'$'),
      endAtLineEnd: true,
      patterns: _docCommentPatterns,
    ),
    TextMateRule(
      id: 'dart.comment.line',
      name: 'comment.line.double-slash.dart',
      type: TokenType.comment,
      match: RegExp(r'//.*$'),
    ),

    // 2. Multiline & Raw Strings
    TextMateRule(
      id: 'dart.string.multiline.raw.double',
      name: 'string.quoted.triple.raw.double.dart',
      type: TokenType.string,
      begin: RegExp(r'r"""'),
      end: RegExp(r'"""'),
    ),
    TextMateRule(
      id: 'dart.string.multiline.raw.single',
      name: 'string.quoted.triple.raw.single.dart',
      type: TokenType.string,
      begin: RegExp(r"r'''"),
      end: RegExp(r"'''"),
    ),
    TextMateRule(
      id: 'dart.string.multiline.double',
      name: 'string.quoted.triple.double.dart',
      type: TokenType.string,
      begin: RegExp(r'"""'),
      end: RegExp(r'"""'),
      patterns: _stringPatterns,
    ),
    TextMateRule(
      id: 'dart.string.multiline.single',
      name: 'string.quoted.triple.single.dart',
      type: TokenType.string,
      begin: RegExp(r"'''"),
      end: RegExp(r"'''"),
      patterns: _stringPatterns,
    ),
    TextMateRule(
      id: 'dart.string.raw.single',
      name: 'string.quoted.raw.single.dart',
      type: TokenType.string,
      match: RegExp(r"r'[^']*'"),
    ),
    TextMateRule(
      id: 'dart.string.raw.double',
      name: 'string.quoted.raw.double.dart',
      type: TokenType.string,
      match: RegExp(r'r"[^"]*"'),
    ),
    TextMateRule(
      id: 'dart.string.single',
      name: 'string.quoted.single.dart',
      type: TokenType.string,
      begin: RegExp(r"'"),
      end: RegExp(r"'"),
      endAtLineEnd: true,
      patterns: _stringPatterns,
    ),
    TextMateRule(
      id: 'dart.string.double',
      name: 'string.quoted.double.dart',
      type: TokenType.string,
      begin: RegExp(r'"'),
      end: RegExp(r'"'),
      endAtLineEnd: true,
      patterns: _stringPatterns,
    ),

    // 3. Annotations
    TextMateRule(
      id: 'dart.annotation',
      name: 'storage.type.annotation.dart',
      type: TokenType.keyword,
      match: RegExp(r'@[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?'),
    ),

    // 4. Constants, Booleans & Language Variables
    TextMateRule(
      id: 'dart.constant.language',
      name: 'constant.language.dart',
      type: TokenType.keyword,
      match: RegExp(r'\b(true|false|null)\b'),
    ),
    TextMateRule(
      id: 'dart.variable.language',
      name: 'variable.language.dart',
      type: TokenType.keyword,
      match: RegExp(r'\b(this|super)\b'),
    ),

    // 5. Control Flow & Directive Keywords
    TextMateRule(
      id: 'dart.keyword.control',
      name: 'keyword.control.dart',
      type: TokenType.keyword,
      match: RegExp(
        r'\b(if|else|switch|case|default|break|continue|return|for|in|while|do|try|catch|on|finally|throw|rethrow|assert|when|yield|async|await|import|export|part\s+of|part|library|show|hide|as|is)\b',
      ),
    ),

    // 6. Declaration & Storage Modifier Keywords
    TextMateRule(
      id: 'dart.keyword.declaration',
      name: 'storage.modifier.dart',
      type: TokenType.keyword,
      match: RegExp(
        r'\b(class|mixin|enum|extension\s+type|extension|typedef|abstract|base|interface|final|sealed|static|const|var|late|required|covariant|external|factory|get|set|operator|with|implements|extends)\b',
      ),
    ),

    // 7. Built-in & Primitive Types
    TextMateRule(
      id: 'dart.type.builtin',
      name: 'storage.type.primitive.dart',
      type: TokenType.identifier,
      match: RegExp(
        r'\b(void|int|double|num|bool|String|dynamic|Object|Never|Null|Future|Stream|Iterable|List|Map|Set|Record|Type|Symbol|Function|DateTime|Duration|Uri|RegExp|BigInt|Pattern|Comparable|Error|Exception|StackTrace)\b',
      ),
    ),

    // 8. Named Argument Labels (e.g. `dtype:`, `title:`, `color:`)
    TextMateRule(
      id: 'dart.parameter.named',
      name: 'variable.parameter.named.dart',
      type: TokenType.identifier,
      match: RegExp(r'\b[a-z_]\w*(?=:(?!=))'),
    ),

    // 9. Function & Method Calls (e.g. `linspace(`, `linspace<Float64>(`, `sin(`)
    TextMateRule(
      id: 'dart.function.call',
      name: 'entity.name.function.dart',
      type: TokenType.identifier,
      match: RegExp(r'\b[a-z_]\w*(?=\s*(?:<[^{};()]*>)?\s*\()'),
    ),

    // 10. UpperCamelCase Classes & Types (including NDArray, Float64, Plot, etc.)
    TextMateRule(
      id: 'dart.type.class',
      name: 'entity.name.type.dart',
      type: TokenType.identifier,
      match: RegExp(r'\b[A-Z][a-zA-Z0-9_]*\b'),
    ),

    // 11. Property / Member Access after `.` (e.g. `.float64`, `.pi`, `.shape`)
    TextMateRule(
      id: 'dart.property.access',
      name: 'variable.other.property.dart',
      type: TokenType.identifier,
      match: RegExp(r'(?<=\.)[a-z_]\w*\b'),
    ),

    // 12. Numbers (Hex, Binary, Float/Scientific, Decimal with `_` separators)
    TextMateRule(
      id: 'dart.number.hex',
      name: 'constant.numeric.hex.dart',
      type: TokenType.number,
      match: RegExp(r'\b0[xX][0-9a-fA-F]+(?:_[0-9a-fA-F]+)*\b'),
    ),
    TextMateRule(
      id: 'dart.number.binary',
      name: 'constant.numeric.binary.dart',
      type: TokenType.number,
      match: RegExp(r'\b0[bB][01]+(?:_[01]+)*\b'),
    ),
    TextMateRule(
      id: 'dart.number.decimal',
      name: 'constant.numeric.decimal.dart',
      type: TokenType.number,
      match: RegExp(
        r'(?:\b\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?|\.\d+(?:_\d+)*)(?:[eE][+-]?\d+(?:_\d+)*)?\b',
      ),
    ),

    // 13. Operators
    TextMateRule(
      id: 'dart.operator',
      name: 'keyword.operator.dart',
      type: TokenType.operator,
      match: RegExp(
        r'(\+\+|--|=>|->|\?\?=?|\?\.|\.\.\.?=?|==|!=|<=|>=|<<=?|>>>?=?|&&|\|\||~/=?|[+\-*/%&|^~!=<>]=?)',
      ),
    ),

    // 14. Punctuation
    TextMateRule(
      id: 'dart.punctuation',
      name: 'punctuation.terminator.dart',
      type: TokenType.punctuation,
      match: RegExp(r'[\(\)\{\}\[\];,.:?]'),
    ),

    // 15. Generic Identifiers / Variables
    TextMateRule(
      id: 'dart.identifier',
      name: 'variable.other.dart',
      type: TokenType.identifier,
      match: RegExp(r'\b[a-zA-Z_$]\w*\b'),
    ),
  ];
}
