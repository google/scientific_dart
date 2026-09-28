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

  /// Complete list of root rules for Dart syntax highlighting.
  static List<TextMateRule> get rules => [
    // 1. Comments
    TextMateRule(
      id: 'dart.comment.block',
      name: 'comment.block.dart',
      type: TokenType.comment,
      begin: RegExp(r'/\*'),
      end: RegExp(r'\*/'),
    ),
    TextMateRule(
      id: 'dart.comment.doc',
      name: 'comment.line.doc.dart',
      type: TokenType.comment,
      match: RegExp(r'///.*$'),
    ),
    TextMateRule(
      id: 'dart.comment.line',
      name: 'comment.line.double-slash.dart',
      type: TokenType.comment,
      match: RegExp(r'//.*$'),
    ),

    // 2. Multiline & Raw Strings
    TextMateRule(
      id: 'dart.string.multiline.double',
      name: 'string.quoted.triple.double.dart',
      type: TokenType.string,
      begin: RegExp(r'r?"""'),
      end: RegExp(r'"""'),
    ),
    TextMateRule(
      id: 'dart.string.multiline.single',
      name: 'string.quoted.triple.single.dart',
      type: TokenType.string,
      begin: RegExp(r"r?'''"),
      end: RegExp(r"'''"),
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
      match: RegExp(r"'(\\.|[^'\\])*'"),
    ),
    TextMateRule(
      id: 'dart.string.double',
      name: 'string.quoted.double.dart',
      type: TokenType.string,
      match: RegExp(r'"(\\.|[^"\\])*"'),
    ),

    // 3. Annotations
    TextMateRule(
      id: 'dart.annotation',
      name: 'meta.declaration.annotation.dart',
      type: TokenType.keyword,
      match: RegExp(r'@[a-zA-Z_]\w*'),
    ),

    // 4. Constants & Booleans
    TextMateRule(
      id: 'dart.constant.language',
      name: 'constant.language.dart',
      type: TokenType.keyword,
      match: RegExp(r'\b(true|false|null)\b'),
    ),

    // 5. Control Flow & Language Keywords
    TextMateRule(
      id: 'dart.keyword.control',
      name: 'keyword.control.dart',
      type: TokenType.keyword,
      match: RegExp(
        r'\b(if|else|switch|case|default|break|continue|return|for|in|while|do|try|catch|on|finally|throw|rethrow|assert|when|yield|async|await)\b',
      ),
    ),
    TextMateRule(
      id: 'dart.keyword.declaration',
      name: 'keyword.declaration.dart',
      type: TokenType.keyword,
      match: RegExp(
        r'\b(class|mixin|enum|extension\s+type|extension|typedef|abstract|base|interface|final|sealed|static|const|var|late|required|covariant|external|factory|get|set|operator|part\s+of|part|library|import|export|show|hide|as|is|with|implements|extends)\b',
      ),
    ),

    // 6. Common Built-in Types (including NDArray & math primitives)
    TextMateRule(
      id: 'dart.type.builtin',
      name: 'support.class.dart',
      type: TokenType.identifier,
      match: RegExp(
        r'\b(int|double|num|bool|String|List|Map|Set|Iterable|Future|Stream|Object|dynamic|void|Never|Record|Type|Symbol|DateTime|Duration|Uri|RegExp|NDArray|DType|Float64|Float32|Int32|Int64|Complex|ScratchArena)\b',
      ),
    ),

    // 7. Numbers (Hex, Float, Int)
    TextMateRule(
      id: 'dart.number.hex',
      name: 'constant.numeric.hex.dart',
      type: TokenType.number,
      match: RegExp(r'\b0[xX][0-9a-fA-F]+\b'),
    ),
    TextMateRule(
      id: 'dart.number.decimal',
      name: 'constant.numeric.decimal.dart',
      type: TokenType.number,
      match: RegExp(r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b'),
    ),

    // 8. Operators
    TextMateRule(
      id: 'dart.operator',
      name: 'keyword.operator.dart',
      type: TokenType.operator,
      match: RegExp(
        r'(\+\+|--|=>|->|\?\?=?|\?\.|\.\.\.?=?|==|!=|<=|>=|<|>|&&|\|\||!|~|\+=|-=|\*=|/=|~/=|%=|&=|\|=|\^=|<<=|>>=|>>>=|[+\-*/%&|^~=])',
      ),
    ),

    // 9. Punctuation
    TextMateRule(
      id: 'dart.punctuation',
      name: 'punctuation.terminator.dart',
      type: TokenType.punctuation,
      match: RegExp(r'[\(\)\{\}\[\];,.:]'),
    ),

    // 10. Generic Identifiers / Method Calls
    TextMateRule(
      id: 'dart.identifier',
      name: 'variable.other.dart',
      type: TokenType.identifier,
      match: RegExp(r'\b[a-zA-Z_]\w*\b'),
    ),
  ];
}
