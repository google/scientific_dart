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

import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Rearranging Examples ===\n');
  runFlipExamples();
  runRollExamples();
}

void runFlipExamples() {
  print('--- 1. Flipping Arrays ---');
  final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  print('Original 2D array:\n$a');
  print('Data: ${a.toList()}\n');

  // Flip all axes
  final flippedAll = flip(a);
  print('Flipped along all axes:\n$flippedAll');
  print('Data: ${flippedAll.toList()}\n');

  // Flip axis 0 (rows)
  final flippedRows = flipud(a);
  print('Flipped along axis 0 (flipud):\n$flippedRows');
  print('Data: ${flippedRows.toList()}\n');

  // Flip axis 1 (columns)
  final flippedCols = fliplr(a);
  print('Flipped along axis 1 (fliplr):\n$flippedCols');
  print('Data: ${flippedCols.toList()}\n');
}

void runRollExamples() {
  print('--- 2. Rolling Arrays ---');
  final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
  print('Original 2D array:\n$a');
  print('Data: ${a.toList()}\n');

  // Roll flat (axis is null)
  final rolledFlat = roll(a, 2);
  print('Rolled flat by 2:\n$rolledFlat');
  print('Data: ${rolledFlat.toList()}\n');

  // Roll along axis 1 (columns)
  final rolledCols = roll(a, 1, axis: 1);
  print('Rolled columns by 1:\n$rolledCols');
  print('Data: ${rolledCols.toList()}\n');

  // Roll along axis 0 (rows)
  final rolledRows = roll(a, 1, axis: 0);
  print('Rolled rows by 1:\n$rolledRows');
  print('Data: ${rolledRows.toList()}\n');
}
