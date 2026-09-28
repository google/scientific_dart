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

import '../core/buffer/text_buffer.dart';
import '../core/history/edit_operation.dart';
import '../core/history/editor_transaction.dart';
import '../core/selection/selection.dart';
import 'lsp_coordinate_translator.dart';
import 'lsp_primitives.dart';

class CodeActionMenuItem {
  final String title;
  final String? kind;
  final bool isPreferred;
  final List<LspTextEdit>? edits;

  CodeActionMenuItem({
    required this.title,
    this.kind,
    this.isPreferred = false,
    this.edits,
  });
}

/// Adapter converting LSP code actions into menu items and core editor transactions.
class LspCodeActionAdapter {
  static List<CodeActionMenuItem> adaptCodeActions(
    List<LspCodeAction> actions,
  ) {
    return actions.map((action) {
      return CodeActionMenuItem(
        title: action.title,
        kind: action.kind,
        isPreferred: action.isPreferred,
        edits: action.edits,
      );
    }).toList();
  }

  /// Converts a list of [LspTextEdit] into a core [EditorTransaction].
  static EditorTransaction editsToTransaction(
    TextBuffer buffer,
    List<LspTextEdit> edits,
    List<Selection> currentSelections,
  ) {
    final ops = <EditOperation>[];

    // Process edits in reverse order so character offsets remain valid
    final sortedEdits = List<LspTextEdit>.from(edits);
    sortedEdits.sort((a, b) => b.range.start.compareTo(a.range.start));

    for (final edit in sortedEdits) {
      final startPos = LspCoordinateTranslator.toTextPosition(
        buffer,
        edit.range.start,
      );
      final endPos = LspCoordinateTranslator.toTextPosition(
        buffer,
        edit.range.end,
      );
      final startOffset = buffer.getOffsetAt(startPos.line, startPos.column);
      final endOffset = buffer.getOffsetAt(endPos.line, endPos.column);
      final deleteLen = endOffset - startOffset;

      if (deleteLen > 0) {
        final deletedText = buffer.getTextInRange(startOffset, deleteLen);
        ops.add(
          DeleteOperation(startOffset, deleteLen, deletedText: deletedText),
        );
      }
      if (edit.newText.isNotEmpty) {
        ops.add(InsertOperation(startOffset, edit.newText));
      }
    }

    return EditorTransaction(
      operations: ops,
      selectionsBefore: currentSelections,
      selectionsAfter: currentSelections,
    );
  }
}
