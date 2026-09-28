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

import '../core/selection/selection_model.dart';
import 'line_wrapping_engine.dart';

/// Information about a single virtual row in the editor viewport layout.
final class VirtualRowInfo {
  /// 0-based virtual row index in the layout.
  final int virtualRowIndex;

  /// Document line index.
  final int lineIndex;

  /// Slice index within the line.
  final int sliceIndex;

  /// The wrapped text slice content.
  final TextLineSlice slice;

  /// Creates a [VirtualRowInfo].
  const VirtualRowInfo({
    required this.virtualRowIndex,
    required this.lineIndex,
    required this.sliceIndex,
    required this.slice,
  });

  /// Whether this row represents the first slice of a document line.
  bool get isFirstSlice => sliceIndex == 0;

  @override
  String toString() =>
      'VirtualRowInfo(vRow: $virtualRowIndex, line: $lineIndex, slice: $sliceIndex, text: "${slice.content}")';
}

/// Virtual position within the editor layout (virtual row and virtual column).
final class VirtualPosition {
  /// 0-based virtual row index.
  final int virtualRow;

  /// 0-based column index within the virtual row slice.
  final int virtualColumn;

  /// Creates a [VirtualPosition].
  const VirtualPosition({
    required this.virtualRow,
    required this.virtualColumn,
  });

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is VirtualPosition &&
        other.virtualRow == virtualRow &&
        other.virtualColumn == virtualColumn;
  }

  @override
  int get hashCode => Object.hash(virtualRow, virtualColumn);

  @override
  String toString() =>
      'VirtualPosition(vRow: $virtualRow, vCol: $virtualColumn)';
}

/// Calculator that maps logical document positions to virtual layout rows with soft wrapping support.
final class VirtualLayoutCalculator {
  /// Line wrapping engine instance.
  final LineWrappingEngine lineWrappingEngine;

  final List<VirtualRowInfo> _virtualRows = [];
  final List<int> _lineToFirstVirtualRow = [];
  final List<List<TextLineSlice>> _lineSlices = [];

  /// Creates a [VirtualLayoutCalculator].
  VirtualLayoutCalculator({
    this.lineWrappingEngine = const LineWrappingEngine(),
  });

  /// Computes the virtual layout mapping for [documentLines].
  void computeLayout(List<String> documentLines) {
    _virtualRows.clear();
    _lineToFirstVirtualRow.clear();
    _lineSlices.clear();

    if (documentLines.isEmpty) {
      final defaultSlice = TextLineSlice(
        lineIndex: 0,
        sliceIndex: 0,
        startColumn: 0,
        endColumn: 0,
        content: '',
      );
      _virtualRows.add(
        VirtualRowInfo(
          virtualRowIndex: 0,
          lineIndex: 0,
          sliceIndex: 0,
          slice: defaultSlice,
        ),
      );
      _lineToFirstVirtualRow.add(0);
      _lineSlices.add([defaultSlice]);
      return;
    }

    int currentVirtualRow = 0;
    for (int lineIdx = 0; lineIdx < documentLines.length; lineIdx++) {
      final lineText = documentLines[lineIdx];
      final slices = lineWrappingEngine.wrapLine(lineText, lineIdx);

      _lineToFirstVirtualRow.add(currentVirtualRow);
      _lineSlices.add(slices);

      for (int sliceIdx = 0; sliceIdx < slices.length; sliceIdx++) {
        _virtualRows.add(
          VirtualRowInfo(
            virtualRowIndex: currentVirtualRow++,
            lineIndex: lineIdx,
            sliceIndex: sliceIdx,
            slice: slices[sliceIdx],
          ),
        );
      }
    }
  }

  /// Total number of virtual rows in the layout.
  int get totalVirtualRows => _virtualRows.length;

  /// Gets row info for virtual row [virtualRow].
  VirtualRowInfo getVirtualRowInfo(int virtualRow) {
    if (_virtualRows.isEmpty) {
      final defaultSlice = TextLineSlice(
        lineIndex: 0,
        sliceIndex: 0,
        startColumn: 0,
        endColumn: 0,
        content: '',
      );
      return VirtualRowInfo(
        virtualRowIndex: 0,
        lineIndex: 0,
        sliceIndex: 0,
        slice: defaultSlice,
      );
    }
    final safeIndex = virtualRow.clamp(0, _virtualRows.length - 1);
    return _virtualRows[safeIndex];
  }

  /// Maps a document [TextPosition] to a [VirtualPosition].
  VirtualPosition documentToVirtualPosition(TextPosition docPosition) {
    if (_lineToFirstVirtualRow.isEmpty) {
      return const VirtualPosition(virtualRow: 0, virtualColumn: 0);
    }

    final safeLine = docPosition.line.clamp(
      0,
      _lineToFirstVirtualRow.length - 1,
    );
    final firstVRow = _lineToFirstVirtualRow[safeLine];
    final slices = _lineSlices[safeLine];

    int targetSliceIndex = 0;
    for (int i = 0; i < slices.length; i++) {
      final s = slices[i];
      if (docPosition.column >= s.startColumn &&
          docPosition.column <= s.endColumn) {
        if (docPosition.column == s.endColumn && i < slices.length - 1) {
          targetSliceIndex = i + 1;
        } else {
          targetSliceIndex = i;
        }
        break;
      }
      if (docPosition.column > s.endColumn) {
        targetSliceIndex = i;
      }
    }

    final matchedSlice = slices[targetSliceIndex];
    final vRow = firstVRow + targetSliceIndex;
    final vCol = (docPosition.column - matchedSlice.startColumn).clamp(
      0,
      matchedSlice.content.length,
    );

    return VirtualPosition(virtualRow: vRow, virtualColumn: vCol);
  }

  /// Maps a [VirtualPosition] back to a document [TextPosition].
  TextPosition virtualToDocumentPosition(VirtualPosition virtualPos) {
    final rowInfo = getVirtualRowInfo(virtualPos.virtualRow);
    final slice = rowInfo.slice;
    final docCol = (slice.startColumn + virtualPos.virtualColumn).clamp(
      slice.startColumn,
      slice.endColumn,
    );
    return TextPosition(rowInfo.lineIndex, docCol);
  }

  /// Gets all slices generated for document line [lineIndex].
  List<TextLineSlice> getSlicesForLine(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= _lineSlices.length) {
      return const [];
    }
    return List.unmodifiable(_lineSlices[lineIndex]);
  }
}
