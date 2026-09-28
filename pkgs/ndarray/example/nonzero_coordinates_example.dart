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
  print('=== NDArray Non-Zero Elements coordinates Extraction ===\n');

  runNonzeroExtractionExample();
}

void runNonzeroExtractionExample() {
  NDArray.scope(() {
    // Allocate a 3x3 grid representing sparse spatial labels/features
    // [[0, 5, 0],
    //  [2, 0, 0],
    //  [0, 0, 9]]
    print('Allocating 3x3 feature matrix:');
    final grid = NDArray.fromList(
      [0.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 9.0],
      [3, 3],
      DType.float64,
    );

    print('Grid:');
    print(grid);

    // nonzero() extracts coordinate indices along each axis:
    // For a 2D array, it returns a list of two 1D NDArrays:
    // indexList[0]: row indices
    // indexList[1]: column indices
    final indices = nonzero(grid);

    final rows = indices[0];
    final cols = indices[1];

    print('\nExtracted Rows Indices: ${rows.toList()}');
    print('Extracted Columns Indices: ${cols.toList()}');

    print('\nMapping coordinates to non-zero element values:');
    final count = rows.shape[0];
    for (var i = 0; i < count; i++) {
      final r = rows.getCell([i]);
      final c = cols.getCell([i]);

      // getCell retrieves the element in-place at the coordinates
      final val = grid.getCell([r, c]);
      print('🏆 Non-Zero Element found at coordinate [$r, $c] -> value: $val');
    }
  });
}
