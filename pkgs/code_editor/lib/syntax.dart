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

/// Syntax highlighting, TextMate rules, scope selectors, and themes.
library;

export 'src/syntax/color_theme.dart';
export 'src/syntax/grammars/dart_grammar.dart';
export 'src/syntax/incremental_tokenizer.dart';
export 'src/syntax/line_state.dart';
export 'src/syntax/scope_matcher.dart';
export 'src/syntax/syntax_token.dart';
export 'src/syntax/syntax_tokenizer.dart';
export 'src/syntax/textmate_lexer.dart';
