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

import 'package:meta/meta.dart';

@immutable
class FoldRegion {
  final int startLine;
  final int endLine;
  final bool isCollapsed;

  const FoldRegion({
    required this.startLine,
    required this.endLine,
    this.isCollapsed = true,
  });

  FoldRegion copyWith({bool? isCollapsed}) {
    return FoldRegion(
      startLine: startLine,
      endLine: endLine,
      isCollapsed: isCollapsed ?? this.isCollapsed,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FoldRegion &&
          runtimeType == other.runtimeType &&
          startLine == other.startLine &&
          endLine == other.endLine &&
          isCollapsed == other.isCollapsed;

  @override
  int get hashCode => Object.hash(startLine, endLine, isCollapsed);

  @override
  String toString() =>
      'FoldRegion($startLine..$endLine, collapsed: $isCollapsed)';
}

class FoldingManager {
  final List<FoldRegion> _regions = [];

  List<FoldRegion> get regions => List.unmodifiable(_regions);

  void addRegion(int startLine, int endLine) {
    if (startLine >= endLine) return;

    final idx = _regions.indexWhere((r) => r.startLine == startLine);
    if (idx != -1) {
      _regions[idx] = FoldRegion(
        startLine: startLine,
        endLine: endLine,
        isCollapsed: _regions[idx].isCollapsed,
      );
    } else {
      _regions.add(
        FoldRegion(startLine: startLine, endLine: endLine, isCollapsed: false),
      );
      _regions.sort((a, b) => a.startLine.compareTo(b.startLine));
    }
  }

  void removeRegion(int startLine) {
    _regions.removeWhere((r) => r.startLine == startLine);
  }

  void collapse(int startLine) {
    _setCollapsed(startLine, true);
  }

  void expand(int startLine) {
    _setCollapsed(startLine, false);
  }

  void toggleFold(int startLine) {
    final r = getRegionAt(startLine);
    if (r != null) {
      _setCollapsed(startLine, !r.isCollapsed);
    }
  }

  void _setCollapsed(int startLine, bool collapsed) {
    final idx = _regions.indexWhere((r) => r.startLine == startLine);
    if (idx != -1) {
      _regions[idx] = _regions[idx].copyWith(isCollapsed: collapsed);
    }
  }

  FoldRegion? getRegionAt(int startLine) {
    for (final r in _regions) {
      if (r.startLine == startLine) return r;
    }
    return null;
  }

  bool isFoldHeader(int lineIndex) {
    return _regions.any((r) => r.startLine == lineIndex);
  }

  bool isLineHidden(int lineIndex) {
    for (final r in _regions) {
      if (r.isCollapsed && lineIndex > r.startLine && lineIndex < r.endLine) {
        return true;
      }
    }
    return false;
  }

  int documentToVisualLine(int docLine, int totalLines) {
    if (docLine <= 0) return 0;
    int visual = 0;
    for (int i = 0; i < docLine && i < totalLines; i++) {
      if (!isLineHidden(i)) {
        visual++;
      }
    }
    return visual;
  }

  int visualToDocumentLine(int visualLine, int totalLines) {
    if (visualLine <= 0) return 0;
    int currentVisual = 0;
    for (int i = 0; i < totalLines; i++) {
      if (!isLineHidden(i)) {
        if (currentVisual == visualLine) return i;
        currentVisual++;
      }
    }
    return totalLines - 1;
  }

  int getVisibleLineCount(int totalLines) {
    int count = 0;
    for (int i = 0; i < totalLines; i++) {
      if (!isLineHidden(i)) count++;
    }
    return count;
  }

  /// Automatically scans document lines for '{' ... '}' blocks and '/*' ... '*/'
  /// block comments, creating or updating fold regions while preserving existing
  /// collapsed states.
  void scanCodeBlocks(List<String> lines) {
    final oldCollapsedMap = <int, bool>{};
    for (final r in _regions) {
      oldCollapsedMap[r.startLine] = r.isCollapsed;
    }
    _regions.clear();

    final stack = <int>[];
    for (int i = 0; i < lines.length; i++) {
      final line = lines[i];
      // Check for opening brace {
      for (int c = 0; c < line.length; c++) {
        final ch = line[c];
        if (ch == '{') {
          stack.add(i);
        } else if (ch == '}') {
          if (stack.isNotEmpty) {
            final start = stack.removeLast();
            if (i > start + 1) {
              final isCollapsed = oldCollapsedMap[start] ?? false;
              _regions.add(
                FoldRegion(
                  startLine: start,
                  endLine: i,
                  isCollapsed: isCollapsed,
                ),
              );
            }
          }
        }
      }
    }

    _regions.sort((a, b) => a.startLine.compareTo(b.startLine));
  }
}
