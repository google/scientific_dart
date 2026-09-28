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

import 'dart:typed_data';
import 'buffer_source.dart';

enum NodeColor { red, black }

class PieceNode {
  BufferType bufferType;
  int start;
  int length;

  NodeColor color;
  PieceNode? parent;
  PieceNode? left;
  PieceNode? right;

  int subtreeLength;
  int subtreeLineCount;
  int lineFeedCount;
  Int32List lineStarts;

  PieceNode({
    required this.bufferType,
    required this.start,
    required this.length,
    required this.lineStarts,
    this.color = NodeColor.red,
  }) : lineFeedCount = lineStarts.length,
       subtreeLength = length,
       subtreeLineCount = lineStarts.length;

  /// Helper to calculate newline relative offsets (`\n`) within a given string [text].
  static Int32List findLineStarts(String text, {int offset = 0, int? length}) {
    final len = length ?? text.length;
    final starts = <int>[];
    for (var i = 0; i < len; i++) {
      if (text.codeUnitAt(offset + i) == 10) {
        // '\n' = 10
        starts.add(i);
      }
    }
    return Int32List.fromList(starts);
  }

  /// Recalculates [subtreeLength] and [subtreeLineCount] from children.
  void updateSubtreeMetadata() {
    var totalLen = length;
    var totalLines = lineFeedCount;

    if (left != null) {
      totalLen += left!.subtreeLength;
      totalLines += left!.subtreeLineCount;
    }
    if (right != null) {
      totalLen += right!.subtreeLength;
      totalLines += right!.subtreeLineCount;
    }

    subtreeLength = totalLen;
    subtreeLineCount = totalLines;
  }
}
