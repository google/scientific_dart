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

import '../buffer/text_buffer.dart';
import '../selection/selection_model.dart';
import 'edit_operation.dart';
import 'editor_transaction.dart';

/// Dual-stack Undo/Redo manager with typing windowing, stack capacity limits,
/// and document version tracking.
class UndoManager {
  final List<EditorTransaction> _undoStack = [];
  final List<EditorTransaction> _redoStack = [];

  final int maxStackSize;
  final Duration typingWindow;

  int _version = 0;
  int _savedVersion = 0;

  UndoManager({
    this.maxStackSize = 100,
    this.typingWindow = const Duration(milliseconds: 1000),
  });

  int get version => _version;
  int get savedVersion => _savedVersion;
  bool get isDirty => _version != _savedVersion;

  bool get canUndo => _undoStack.isNotEmpty;
  bool get canRedo => _redoStack.isNotEmpty;

  void markSaved() {
    _savedVersion = _version;
  }

  /// Commits a new transaction to the undo stack.
  void commitTransaction(EditorTransaction tx) {
    if (tx.operations.isEmpty) return;

    _redoStack.clear();
    _version++;

    // Check typing window coalescing
    if (_undoStack.isNotEmpty) {
      final lastTx = _undoStack.last;
      if (_canCoalesce(lastTx, tx)) {
        final mergedOps = List<EditOperation>.from(lastTx.operations)
          ..addAll(tx.operations);
        _undoStack[_undoStack.length - 1] = EditorTransaction(
          operations: mergedOps,
          selectionsBefore: lastTx.selectionsBefore,
          selectionsAfter: tx.selectionsAfter,
          timestamp: tx.timestamp,
        );
        return;
      }
    }

    _undoStack.add(tx);
    if (_undoStack.length > maxStackSize) {
      _undoStack.removeAt(0);
    }
  }

  bool _canCoalesce(EditorTransaction last, EditorTransaction current) {
    final diff = current.timestamp.difference(last.timestamp);
    if (diff > typingWindow) return false;

    if (last.operations.length == 1 && current.operations.length == 1) {
      final op1 = last.operations.first;
      final op2 = current.operations.first;

      if (op1 is InsertOperation && op2 is InsertOperation) {
        // Coalesce sequential single-character insertions (avoid whitespace splitting or large multi-character inserts)
        if (op1.text.length == 1 &&
            op2.text.length == 1 &&
            op2.offset == op1.offset + op1.text.length) {
          return op1.text != '\n' && op2.text != '\n';
        }
      }
    }
    return false;
  }

  /// Executes undo and applies inverted operations to [buffer] and [selectionModel].
  EditorTransaction? undo(TextBuffer buffer, SelectionModel selectionModel) {
    if (!canUndo) return null;

    final tx = _undoStack.removeLast();
    final inverted = tx.invert();

    for (final op in inverted.operations) {
      op.apply(buffer);
    }

    selectionModel.setSelections(inverted.selectionsAfter);
    _redoStack.add(tx);
    _version--;

    return inverted;
  }

  /// Executes redo and reapplies operations to [buffer] and [selectionModel].
  EditorTransaction? redo(TextBuffer buffer, SelectionModel selectionModel) {
    if (!canRedo) return null;

    final tx = _redoStack.removeLast();

    for (final op in tx.operations) {
      op.apply(buffer);
    }

    selectionModel.setSelections(tx.selectionsAfter);
    _undoStack.add(tx);
    _version++;

    return tx;
  }

  void clear() {
    _undoStack.clear();
    _redoStack.clear();
    _version = 0;
    _savedVersion = 0;
  }
}
