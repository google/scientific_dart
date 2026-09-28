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

import 'buffer_source.dart';
import 'piece_node.dart';
import 'text_buffer.dart';

class PieceTreeTextBuffer implements TextBuffer {
  final OriginalBuffer originalBuffer;
  final AddBuffer addBuffer = AddBuffer();
  PieceNode? root;

  PieceTreeTextBuffer(String initialContent)
    : originalBuffer = OriginalBuffer(initialContent) {
    if (initialContent.isNotEmpty) {
      final lineStarts = PieceNode.findLineStarts(initialContent);
      root = PieceNode(
        bufferType: BufferType.original,
        start: 0,
        length: initialContent.length,
        lineStarts: lineStarts,
        color: NodeColor.black,
      );
    }
  }

  @override
  int get length => root?.subtreeLength ?? 0;

  @override
  int get lineCount => (root?.subtreeLineCount ?? 0) + 1;

  @override
  String get text {
    if (root == null) return '';
    return getTextInRange(0, length);
  }

  /// Returns text content split into document lines.
  List<String> get lines => text.split('\n');

  @override
  String getTextInRange(int startOffset, int reqLength) {
    if (reqLength <= 0 || root == null) return '';
    final start = startOffset.clamp(0, length);
    final len = reqLength.clamp(0, length - start);
    if (len <= 0) return '';

    final sb = StringBuffer();
    _collectText(root, start, len, sb);
    return sb.toString();
  }

  void _collectText(
    PieceNode? node,
    int offset,
    int remainingLength,
    StringBuffer sb,
  ) {
    if (node == null || remainingLength <= 0) return;

    final leftLength = node.left?.subtreeLength ?? 0;

    if (offset < leftLength) {
      // Needs text from left subtree
      final leftTake = (leftLength - offset).clamp(0, remainingLength);
      _collectText(node.left, offset, leftTake, sb);
      remainingLength -= leftTake;
      offset = 0;
    } else {
      offset -= leftLength;
    }

    if (remainingLength <= 0) return;

    if (offset < node.length) {
      // Needs text from this node
      final nodeTake = (node.length - offset).clamp(0, remainingLength);
      final pieceText = _getNodeText(node, node.start + offset, nodeTake);
      sb.write(pieceText);
      remainingLength -= nodeTake;
      offset = 0;
    } else {
      offset -= node.length;
    }

    if (remainingLength <= 0) return;

    // Needs text from right subtree
    _collectText(node.right, offset, remainingLength, sb);
  }

  String _getNodeText(PieceNode node, int start, int length) {
    if (node.bufferType == BufferType.original) {
      return originalBuffer.content.substring(start, start + length);
    } else {
      return addBuffer.getText(start, length);
    }
  }

  @override
  String getLine(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= lineCount) {
      throw RangeError.range(lineIndex, 0, lineCount - 1, 'lineIndex');
    }
    final start = getLineOffset(lineIndex);
    final end = (lineIndex + 1 < lineCount)
        ? getLineOffset(lineIndex + 1)
        : length;
    return getTextInRange(start, end - start);
  }

  @override
  int getLineOffset(int lineIndex) {
    if (lineIndex <= 0) return 0;
    if (lineIndex >= lineCount) return length;

    // Line L (1-based index L = lineIndex) starts right after the L-th '\n' in the tree.
    return _findNthLineFeedOffset(root, lineIndex) + 1;
  }

  int _findNthLineFeedOffset(PieceNode? node, int n) {
    if (node == null || n <= 0) return 0;

    final leftLines = node.left?.subtreeLineCount ?? 0;
    if (n <= leftLines) {
      return _findNthLineFeedOffset(node.left, n);
    }

    n -= leftLines;
    final leftLength = node.left?.subtreeLength ?? 0;

    if (n <= node.lineFeedCount) {
      final relLineFeed = node.lineStarts[n - 1];
      return leftLength + relLineFeed;
    }

    n -= node.lineFeedCount;
    return leftLength + node.length + _findNthLineFeedOffset(node.right, n);
  }

  @override
  int getLineLength(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= lineCount) return 0;
    final start = getLineOffset(lineIndex);
    final end = (lineIndex + 1 < lineCount)
        ? getLineOffset(lineIndex + 1)
        : length;
    return end - start;
  }

  @override
  (int line, int column) getLineAndColumnAt(int offset) {
    final targetOffset = offset.clamp(0, length);
    if (root == null || targetOffset == 0) return (0, 0);

    final line = _countLineFeedsBeforeOffset(root, targetOffset);
    final lineStartOffset = getLineOffset(line);
    final col = targetOffset - lineStartOffset;
    return (line, col);
  }

  int _countLineFeedsBeforeOffset(PieceNode? node, int offset) {
    if (node == null || offset <= 0) return 0;

    final leftLength = node.left?.subtreeLength ?? 0;
    if (offset <= leftLength) {
      return _countLineFeedsBeforeOffset(node.left, offset);
    }

    var totalLines = (node.left?.subtreeLineCount ?? 0);
    final relOffset = offset - leftLength;

    if (relOffset < node.length) {
      for (var i = 0; i < node.lineFeedCount; i++) {
        if (node.lineStarts[i] < relOffset) {
          totalLines++;
        } else {
          break;
        }
      }
      return totalLines;
    }

    totalLines += node.lineFeedCount;
    return totalLines +
        _countLineFeedsBeforeOffset(node.right, relOffset - node.length);
  }

  @override
  int getOffsetAt(int line, int column) {
    final lineIdx = line.clamp(0, lineCount - 1);
    final start = getLineOffset(lineIdx);
    final lineLen = getLineLength(lineIdx);
    final col = column.clamp(0, lineLen);
    return start + col;
  }

  /// Adjusts [offset] to avoid breaking UTF-16 surrogate pairs (`0xD800`..`0xDBFF` followed by `0xDC00`..`0xDFFF`).
  int _adjustOffsetForSurrogatePair(int offset) {
    if (offset <= 0 || offset >= length) return offset;
    final full = text;
    final codePrev = full.codeUnitAt(offset - 1);
    final codeCurr = full.codeUnitAt(offset);

    if (codePrev >= 0xD800 &&
        codePrev <= 0xDBFF &&
        codeCurr >= 0xDC00 &&
        codeCurr <= 0xDFFF) {
      // Split lands inside surrogate pair, move after low surrogate
      return offset + 1;
    }
    return offset;
  }

  @override
  void insert(int offset, String newText) {
    if (newText.isEmpty) return;
    var safeOffset = _adjustOffsetForSurrogatePair(offset);
    safeOffset = safeOffset.clamp(0, length);

    final addStart = addBuffer.append(newText);
    final lineStarts = PieceNode.findLineStarts(newText);
    final newNode = PieceNode(
      bufferType: BufferType.add,
      start: addStart,
      length: newText.length,
      lineStarts: lineStarts,
    );

    if (root == null) {
      root = newNode..color = NodeColor.black;
      return;
    }

    _splitAndInsertAt(safeOffset, newNode);
  }

  @override
  void delete(int offset, int deleteLength) {
    if (deleteLength <= 0 || root == null) return;
    final startOffset = _adjustOffsetForSurrogatePair(offset).clamp(0, length);
    final endOffset = _adjustOffsetForSurrogatePair(
      offset + deleteLength,
    ).clamp(0, length);

    if (endOffset <= startOffset) return;

    // Convert piece tree to pieces list, slice out range, and rebuild balanced tree
    final pieces = _toPieceList();
    final newPieces = <PieceNode>[];

    var currentOffset = 0;
    for (final p in pieces) {
      final pStart = currentOffset;
      final pEnd = currentOffset + p.length;

      if (pEnd <= startOffset || pStart >= endOffset) {
        // Outside delete range
        newPieces.add(p);
      } else {
        // Overlaps delete range
        if (startOffset > pStart) {
          // Keep left chunk
          final leftLen = startOffset - pStart;
          final chunkText = _getNodeText(p, p.start, leftLen);
          newPieces.add(
            PieceNode(
              bufferType: p.bufferType,
              start: p.start,
              length: leftLen,
              lineStarts: PieceNode.findLineStarts(chunkText),
            ),
          );
        }
        if (endOffset < pEnd) {
          // Keep right chunk
          final rightSkip = endOffset - pStart;
          final rightLen = pEnd - endOffset;
          final chunkText = _getNodeText(p, p.start + rightSkip, rightLen);
          newPieces.add(
            PieceNode(
              bufferType: p.bufferType,
              start: p.start + rightSkip,
              length: rightLen,
              lineStarts: PieceNode.findLineStarts(chunkText),
            ),
          );
        }
      }
      currentOffset = pEnd;
    }

    _rebuildFromPieces(_coalescePieces(newPieces));
  }

  void _splitAndInsertAt(int offset, PieceNode newNode) {
    final pieces = _toPieceList();
    final newPieces = <PieceNode>[];

    var currentOffset = 0;
    var inserted = false;

    for (final p in pieces) {
      final pStart = currentOffset;
      final pEnd = currentOffset + p.length;

      if (!inserted && offset >= pStart && offset <= pEnd) {
        final leftLen = offset - pStart;
        if (leftLen > 0) {
          final leftText = _getNodeText(p, p.start, leftLen);
          newPieces.add(
            PieceNode(
              bufferType: p.bufferType,
              start: p.start,
              length: leftLen,
              lineStarts: PieceNode.findLineStarts(leftText),
            ),
          );
        }

        newPieces.add(newNode);

        final rightLen = pEnd - offset;
        if (rightLen > 0) {
          final rightText = _getNodeText(p, p.start + leftLen, rightLen);
          newPieces.add(
            PieceNode(
              bufferType: p.bufferType,
              start: p.start + leftLen,
              length: rightLen,
              lineStarts: PieceNode.findLineStarts(rightText),
            ),
          );
        }

        inserted = true;
      } else {
        newPieces.add(p);
      }
      currentOffset = pEnd;
    }

    if (!inserted) {
      newPieces.add(newNode);
    }

    _rebuildFromPieces(_coalescePieces(newPieces));
  }

  List<PieceNode> _toPieceList() {
    final list = <PieceNode>[];
    void traverse(PieceNode? node) {
      if (node == null) return;
      traverse(node.left);
      list.add(
        PieceNode(
          bufferType: node.bufferType,
          start: node.start,
          length: node.length,
          lineStarts: node.lineStarts,
        ),
      );
      traverse(node.right);
    }

    traverse(root);
    return list;
  }

  List<PieceNode> _coalescePieces(List<PieceNode> pieces) {
    if (pieces.isEmpty) return pieces;
    final result = <PieceNode>[];
    var current = pieces.first;

    for (var i = 1; i < pieces.length; i++) {
      final next = pieces[i];
      if (current.bufferType == next.bufferType &&
          current.start + current.length == next.start) {
        // Merge contiguous pieces
        final mergedText = _getNodeText(
          current,
          current.start,
          current.length + next.length,
        );
        current = PieceNode(
          bufferType: current.bufferType,
          start: current.start,
          length: current.length + next.length,
          lineStarts: PieceNode.findLineStarts(mergedText),
        );
      } else {
        result.add(current);
        current = next;
      }
    }
    result.add(current);
    return result;
  }

  void _rebuildFromPieces(List<PieceNode> pieces) {
    if (pieces.isEmpty) {
      root = null;
      return;
    }

    PieceNode buildBalancedTree(
      List<PieceNode> nodes,
      int start,
      int end,
      PieceNode? parent,
    ) {
      final mid = (start + end) ~/ 2;
      final node = nodes[mid];
      node.parent = parent;
      node.color = NodeColor.black;

      if (start < mid) {
        node.left = buildBalancedTree(nodes, start, mid - 1, node);
      } else {
        node.left = null;
      }

      if (end > mid) {
        node.right = buildBalancedTree(nodes, mid + 1, end, node);
      } else {
        node.right = null;
      }

      node.updateSubtreeMetadata();
      return node;
    }

    root = buildBalancedTree(pieces, 0, pieces.length - 1, null);
  }
}
