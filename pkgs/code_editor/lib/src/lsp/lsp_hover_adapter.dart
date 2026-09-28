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

import 'lsp_primitives.dart';

class HoverTooltipViewModel {
  final String markdownContent;
  final LspRange? range;

  HoverTooltipViewModel({required this.markdownContent, this.range});
}

/// Adapter presenting LSP hover information as Markdown tooltip view models.
class LspHoverAdapter {
  static HoverTooltipViewModel? adaptHover(LspHover? hover) {
    if (hover == null || hover.contents.isEmpty) return null;
    return HoverTooltipViewModel(
      markdownContent: hover.contents,
      range: hover.range,
    );
  }
}
