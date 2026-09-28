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

import 'line_state.dart';
import 'syntax_token.dart';
import 'syntax_tokenizer.dart';

class TextMateRule {
  final String id;
  final String? name;
  final TokenType type;
  final RegExp? match;
  final RegExp? begin;
  final RegExp? end;
  final List<TextMateRule> patterns;

  TextMateRule({
    required this.id,
    this.name,
    this.type = TokenType.custom,
    this.match,
    this.begin,
    this.end,
    this.patterns = const [],
  });

  bool get isBeginEnd => begin != null && end != null;
}

class TextMateLexer implements SyntaxTokenizer {
  final List<TextMateRule> rootRules;
  final Map<String, TextMateRule> ruleRegistry = {};

  TextMateLexer({required this.rootRules}) {
    _registerRules(rootRules);
  }

  void _registerRules(List<TextMateRule> rules) {
    for (final rule in rules) {
      ruleRegistry[rule.id] = rule;
      if (rule.patterns.isNotEmpty) {
        _registerRules(rule.patterns);
      }
    }
  }

  @override
  LineTokenizationResult tokenizeLine(
    String lineText,
    LineState previousState,
  ) {
    final stack = <String>[];
    if (previousState is StackLineState) {
      stack.addAll(previousState.stack);
    }

    final tokens = <SyntaxToken>[];
    int offset = 0;

    while (offset < lineText.length) {
      final activeRule = stack.isNotEmpty ? ruleRegistry[stack.last] : null;
      bool matched = false;

      // 1. Check if active begin/end rule matches 'end' pattern
      if (activeRule != null && activeRule.end != null) {
        final match = activeRule.end!.matchAsPrefix(lineText, offset);
        if (match != null) {
          final matchedText = match.group(0)!;
          final scopes = stack
              .map((id) => ruleRegistry[id]?.name)
              .whereType<String>()
              .map((s) => StyleScope(s))
              .toList();

          tokens.add(
            SyntaxToken(
              offset: offset,
              length: matchedText.length,
              type: activeRule.type,
              scopes: scopes,
              text: matchedText,
            ),
          );

          offset += matchedText.length;
          stack.removeLast();
          matched = true;
          continue;
        }
      }

      // 2. Check rules available in current scope (nested rules or root rules)
      final availableRules =
          activeRule != null && activeRule.patterns.isNotEmpty
          ? activeRule.patterns
          : rootRules;

      for (final rule in availableRules) {
        if (rule.isBeginEnd) {
          final match = rule.begin!.matchAsPrefix(lineText, offset);
          if (match != null) {
            final matchedText = match.group(0)!;
            stack.add(rule.id);
            final scopes = stack
                .map((id) => ruleRegistry[id]?.name)
                .whereType<String>()
                .map((s) => StyleScope(s))
                .toList();

            tokens.add(
              SyntaxToken(
                offset: offset,
                length: matchedText.length,
                type: rule.type,
                scopes: scopes,
                text: matchedText,
              ),
            );

            offset += matchedText.length;
            matched = true;
            break;
          }
        } else if (rule.match != null) {
          final match = rule.match!.matchAsPrefix(lineText, offset);
          if (match != null) {
            final matchedText = match.group(0)!;
            final scopes = [
              ...stack
                  .map((id) => ruleRegistry[id]?.name)
                  .whereType<String>()
                  .map((s) => StyleScope(s)),
              if (rule.name != null) StyleScope(rule.name!),
            ];

            tokens.add(
              SyntaxToken(
                offset: offset,
                length: matchedText.length,
                type: rule.type,
                scopes: scopes,
                text: matchedText,
              ),
            );

            offset += matchedText.length;
            matched = true;
            break;
          }
        }
      }

      if (matched) continue;

      // 3. Fallback: single character unmatched token
      final scopes = stack
          .map((id) => ruleRegistry[id]?.name)
          .whereType<String>()
          .map((s) => StyleScope(s))
          .toList();

      tokens.add(
        SyntaxToken(
          offset: offset,
          length: 1,
          type: TokenType.unknown,
          scopes: scopes,
          text: lineText[offset],
        ),
      );
      offset++;
    }

    final coalesced = _coalesceTokens(tokens);

    return LineTokenizationResult(
      tokens: coalesced,
      endState: stack.isEmpty ? const EmptyLineState() : StackLineState(stack),
    );
  }

  List<SyntaxToken> _coalesceTokens(List<SyntaxToken> tokens) {
    if (tokens.isEmpty) return tokens;
    final result = <SyntaxToken>[];
    SyntaxToken current = tokens.first;

    for (int i = 1; i < tokens.length; i++) {
      final next = tokens[i];
      if (current.type == next.type &&
          _scopesEqual(current.scopes, next.scopes) &&
          current.end == next.offset) {
        current = SyntaxToken(
          offset: current.offset,
          length: current.length + next.length,
          type: current.type,
          scopes: current.scopes,
          text: current.text + next.text,
        );
      } else {
        result.add(current);
        current = next;
      }
    }
    result.add(current);
    return result;
  }

  bool _scopesEqual(List<StyleScope> a, List<StyleScope> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
