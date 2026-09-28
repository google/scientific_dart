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

/// Core code_editor library export.
library;

export 'core.dart';
export 'lsp.dart';
export 'render.dart';
export 'syntax.dart';
export 'viewport.dart';
export 'src/editor/code_editor_controller.dart';
export 'src/editor/editor_options.dart';
export 'src/features/bracket_matching/bracket_matcher.dart';
export 'src/features/find_replace/find_replace_controller.dart';
export 'src/features/find_replace/search_match.dart';
export 'src/features/smart_editing/auto_close_pairs.dart';
export 'src/features/smart_editing/line_operations.dart';
export 'src/features/smart_editing/smart_indent_engine.dart';
export 'src/features/formatting/dart_formatter_engine.dart';
export 'src/features/snippets/snippet_engine.dart';
export 'src/lsp/lsp_diagnostic_adapter.dart';
export 'src/syntax/grammars/dart_grammar.dart';
