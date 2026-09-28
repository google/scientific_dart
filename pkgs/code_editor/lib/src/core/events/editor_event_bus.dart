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

import 'dart:async';
import '../history/editor_transaction.dart';
import '../selection/selection.dart';
import '../state/document_snapshot.dart';

class DocumentChangeEvent {
  final EditorTransaction transaction;
  final DocumentSnapshot snapshot;

  DocumentChangeEvent(this.transaction, this.snapshot);
}

class SelectionChangeEvent {
  final List<Selection> selections;
  final Selection primary;

  SelectionChangeEvent(this.selections, this.primary);
}

class HistoryChangeEvent {
  final bool canUndo;
  final bool canRedo;
  final int version;

  HistoryChangeEvent({
    required this.canUndo,
    required this.canRedo,
    required this.version,
  });
}

/// Reactive event bus exposing document, selection, and history streams.
class EditorEventBus {
  final _documentChangeController =
      StreamController<DocumentChangeEvent>.broadcast();
  final _selectionChangeController =
      StreamController<SelectionChangeEvent>.broadcast();
  final _historyChangeController =
      StreamController<HistoryChangeEvent>.broadcast();

  Stream<DocumentChangeEvent> get onDocumentChange =>
      _documentChangeController.stream;
  Stream<SelectionChangeEvent> get onSelectionChange =>
      _selectionChangeController.stream;
  Stream<HistoryChangeEvent> get onHistoryChange =>
      _historyChangeController.stream;

  void emitDocumentChange(DocumentChangeEvent event) {
    _documentChangeController.add(event);
  }

  void emitSelectionChange(SelectionChangeEvent event) {
    _selectionChangeController.add(event);
  }

  void emitHistoryChange(HistoryChangeEvent event) {
    _historyChangeController.add(event);
  }

  void dispose() {
    unawaited(_documentChangeController.close());
    unawaited(_selectionChangeController.close());
    unawaited(_historyChangeController.close());
  }
}
