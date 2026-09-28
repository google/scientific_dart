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

/// Result of tokenizing a single line of code.
class LineTokenizationResult {
  final List<SyntaxToken> tokens;
  final LineState endState;

  const LineTokenizationResult({required this.tokens, required this.endState});
}

/// Abstract line tokenization interface.
abstract class SyntaxTokenizer {
  /// Tokenizes a single line of text starting with [previousState].
  LineTokenizationResult tokenizeLine(String lineText, LineState previousState);
}
