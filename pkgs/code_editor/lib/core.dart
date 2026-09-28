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

/// Core text buffer, transaction history, and multi-cursor selection engine.
///
/// Includes piece tree text buffer, undo/redo manager, atomic transactions,
/// selection models, and reactive snapshot event streams.
library;

export 'src/core/buffer/buffer_source.dart';
export 'src/core/buffer/piece_node.dart';
export 'src/core/buffer/piece_tree.dart';
export 'src/core/buffer/text_buffer.dart';
export 'src/core/events/editor_event_bus.dart';
export 'src/core/events/ime_input_handler.dart';
export 'src/core/events/keyboard_navigation_handler.dart';
export 'src/core/events/mouse_selection_handler.dart';
export 'src/core/history/edit_operation.dart';
export 'src/core/history/editor_transaction.dart';
export 'src/core/history/undo_manager.dart';
export 'src/core/selection/selection_model.dart';
export 'src/core/state/document_snapshot.dart';
