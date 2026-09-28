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

import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('LineHeightTree', () {
    test('computes O(log N) prefix sums correctly', () {
      final tree = LineHeightTree(5, defaultHeight: 20.0);
      expect(tree.length, equals(5));
      expect(tree.totalHeight, equals(100.0));

      expect(tree.prefixSum(0), equals(20.0));
      expect(tree.prefixSum(1), equals(40.0));
      expect(tree.prefixSum(2), equals(60.0));
      expect(tree.prefixSum(4), equals(100.0));

      expect(tree.getLineTop(0), equals(0.0));
      expect(tree.getLineBottom(0), equals(20.0));
      expect(tree.getLineTop(2), equals(40.0));
      expect(tree.getLineBottom(2), equals(60.0));
    });

    test('updates line height dynamically and recalculates prefix sums', () {
      final tree = LineHeightTree(5, defaultHeight: 20.0);
      tree.setLineHeight(1, 50.0);

      expect(tree.totalHeight, equals(130.0));
      expect(tree.prefixSum(0), equals(20.0));
      expect(tree.prefixSum(1), equals(70.0));
      expect(tree.prefixSum(2), equals(90.0));
    });

    test('lineAtHeight performs O(log N) binary search mapping correctly', () {
      final tree = LineHeightTree(4, defaultHeight: 20.0);

      expect(tree.lineAtHeight(0.0), equals(0));
      expect(tree.lineAtHeight(10.0), equals(0));
      expect(tree.lineAtHeight(19.99), equals(0));
      expect(tree.lineAtHeight(20.0), equals(1));
      expect(tree.lineAtHeight(35.0), equals(1));
      expect(tree.lineAtHeight(65.0), equals(3));
      expect(tree.lineAtHeight(100.0), equals(3));
    });

    test('handles resizing for dynamic document line counts', () {
      final tree = LineHeightTree(3, defaultHeight: 20.0);
      expect(tree.totalHeight, equals(60.0));

      tree.resize(6, defaultHeight: 30.0);
      expect(tree.length, equals(6));
      expect(tree.totalHeight, equals(150.0));
      expect(tree.getLineHeight(0), equals(20.0));
      expect(tree.getLineHeight(3), equals(30.0));
    });
  });
}
