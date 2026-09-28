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

/// O(log N) prefix-sum spatial tree for variable visual line heights (soft wraps & folds)
/// using packed Float64List arrays for dart2wasm performance.
class LineHeightTree {
  Float64List _heights;
  Float64List _tree;
  int _length;

  LineHeightTree(int lineCount, {double defaultHeight = 20.0})
    : _length = lineCount,
      _heights = Float64List(lineCount),
      _tree = Float64List(lineCount + 1) {
    if (lineCount > 0) {
      for (int i = 0; i < lineCount; i++) {
        _heights[i] = defaultHeight;
      }
      _rebuildTree();
    }
  }

  int get length => _length;

  double get totalHeight => _length > 0 ? prefixSum(_length - 1) : 0.0;

  double getLineHeight(int index) {
    if (index < 0 || index >= _length) return 0.0;
    return _heights[index];
  }

  void setLineHeight(int index, double height) {
    if (index < 0 || index >= _length) return;
    final delta = height - _heights[index];
    if (delta == 0.0) return;

    _heights[index] = height;
    int idx = index + 1;
    while (idx <= _length) {
      _tree[idx] += delta;
      idx += idx & -idx;
    }
  }

  /// Calculates cumulative prefix sum of heights from line 0 to [index] (inclusive). O(log N).
  double prefixSum(int index) {
    if (index < 0) return 0.0;
    if (index >= _length) index = _length - 1;

    double sum = 0.0;
    int idx = index + 1;
    while (idx > 0) {
      sum += _tree[idx];
      idx -= idx & -idx;
    }
    return sum;
  }

  /// Calculates pixel Y top position of line [index].
  double getLineTop(int index) {
    if (index <= 0) return 0.0;
    return prefixSum(index - 1);
  }

  /// Calculates pixel Y bottom position of line [index].
  double getLineBottom(int index) {
    if (index < 0) return 0.0;
    return prefixSum(index);
  }

  /// Finds line index corresponding to cumulative Y pixel position using O(log N) binary lifting.
  int lineAtHeight(double pixelY) {
    if (_length <= 0 || pixelY <= 0.0) return 0;

    int idx = 0;
    double currentSum = 0.0;

    int mask = 1;
    while ((mask << 1) <= _length) {
      mask <<= 1;
    }

    for (int step = mask; step > 0; step >>= 1) {
      final int nextIdx = idx + step;
      if (nextIdx <= _length) {
        if (currentSum + _tree[nextIdx] <= pixelY) {
          idx = nextIdx;
          currentSum += _tree[idx];
        }
      }
    }

    if (idx >= _length) return _length - 1;
    return idx;
  }

  void _rebuildTree() {
    _tree.fillRange(0, _length + 1, 0.0);
    for (int i = 0; i < _length; i++) {
      final int idx = i + 1;
      _tree[idx] += _heights[i];
      final int parent = idx + (idx & -idx);
      if (parent <= _length) {
        _tree[parent] += _tree[idx];
      }
    }
  }

  /// Resizes tree for new line count.
  void resize(int newCount, {double defaultHeight = 20.0}) {
    if (newCount == _length) return;

    final newHeights = Float64List(newCount);
    final minLen = newCount < _length ? newCount : _length;

    for (int i = 0; i < minLen; i++) {
      newHeights[i] = _heights[i];
    }
    for (int i = minLen; i < newCount; i++) {
      newHeights[i] = defaultHeight;
    }

    _length = newCount;
    _heights = newHeights;
    _tree = Float64List(newCount + 1);
    _rebuildTree();
  }
}
