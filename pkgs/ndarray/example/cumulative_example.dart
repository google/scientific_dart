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
  print(
    '=== NDArray cumsum() and cumprod() Cumulative Extractions Examples ===\n',
  );

  NDArray.scope(() {
    // 1. Flat sequence cumulative sum
    final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
    print('Original flat array:');
    print(' [ ${a.toList().join(", ")} ]');

    final csFlat = cumsum(a);
    print('\nFlat cumulative sum (cumsum):');
    print(' [ ${csFlat.toList().join(", ")} ]');

    // 2. Multi-dimensional cumulative sum along axes
    final mat = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );

    print('\nOriginal 2D Matrix (2x3):');
    _printMatrix(mat);

    final csRow = cumsum(mat, axis: 0);
    print('\nCumulative sum along columns (axis=0):');
    _printMatrix(csRow);

    final csCol = cumsum(mat, axis: 1);
    print('\nCumulative sum along rows (axis=1):');
    _printMatrix(csCol);

    // 3. Cumulative product along axis
    final cpCol = cumprod(mat, axis: 1);
    print('\nCumulative product along rows (axis=1):');
    _printMatrix(cpCol);

    // 4. In-place recycler buffer reuse
    final recycler = NDArray<double>.zeros([2, 3], DType.float64);
    final recycled = cumsum(mat, axis: 0, out: recycler);
    print(
      '\nRecycled buffer (identical check): ${identical(recycled, recycler) ? "PASS" : "FAIL"}',
    );
    _printMatrix(recycled);
  });
}

void _printMatrix(NDArray a) {
  final rows = a.shape[0];
  final cols = a.shape[1];
  for (var r = 0; r < rows; r++) {
    final rowStr = [];
    for (var c = 0; c < cols; c++) {
      rowStr.add(a[r * cols + c].toStringAsFixed(1).padLeft(5));
    }
    print(' [ ${rowStr.join(", ")} ]');
  }
}
