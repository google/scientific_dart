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

/// Immutable snapshot of editor document state at a specific version.
class DocumentSnapshot {
  final String content;
  final int version;
  final int lineCount;
  final List<Selection> selections;
  final DateTime timestamp;

  DocumentSnapshot({
    required this.content,
    required this.version,
    required this.lineCount,
    required List<Selection> selections,
    DateTime? timestamp,
  }) : selections = List.unmodifiable(selections),
       timestamp = timestamp ?? DateTime.now();

  @override
  String toString() =>
      'DocumentSnapshot(v$version, lines: $lineCount, length: ${content.length})';
}
