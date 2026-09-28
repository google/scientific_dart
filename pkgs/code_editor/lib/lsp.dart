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

/// Language Server Protocol (LSP) 3.17 integration adapters.
///
/// Provides UTF-16 coordinate translation, incremental document sync manager,
/// and feature adapters for completions, diagnostics, hover tooltips, symbols, and code actions.
library;

export 'src/lsp/completion_popup_model.dart';
export 'src/lsp/hover_tooltip_model.dart';
export 'src/lsp/lsp_code_action_adapter.dart';
export 'src/lsp/lsp_completion_adapter.dart';
export 'src/lsp/lsp_coordinate_translator.dart';
export 'src/lsp/lsp_diagnostic_adapter.dart';
export 'src/lsp/lsp_hover_adapter.dart';
export 'src/lsp/lsp_primitives.dart';
export 'src/lsp/lsp_symbol_adapter.dart';
export 'src/lsp/lsp_sync_manager.dart';
