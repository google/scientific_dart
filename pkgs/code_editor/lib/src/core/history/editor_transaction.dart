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

import '../selection/selection.dart';
import 'edit_operation.dart';

/// Represents an atomic batch of edit operations on a document text buffer.
class EditorTransaction {
  final List<EditOperation> operations;
  final List<Selection> selectionsBefore;
  final List<Selection> selectionsAfter;
  final DateTime timestamp;

  EditorTransaction({
    required this.operations,
    required this.selectionsBefore,
    required this.selectionsAfter,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// Inverts the transaction for undo execution.
  EditorTransaction invert() {
    final invertedOps = operations.reversed.map((op) => op.invert()).toList();
    return EditorTransaction(
      operations: invertedOps,
      selectionsBefore: selectionsAfter,
      selectionsAfter: selectionsBefore,
      timestamp: DateTime.now(),
    );
  }
}
