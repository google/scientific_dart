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
  print('=== NDArray tril() and triu() Triangular Extractions Examples ===\n');

  NDArray.scope(() {
    // 1. Create a 3x3 matrix
    final a = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      [3, 3],
      DType.float64,
    );

    print('Original 3x3 Matrix:');
    _printMatrix(a);

    // 2. tril: Lower triangular extraction (k = 0)
    final lower = tril(a);
    print('\nLower Triangular (tril, k=0):');
    _printMatrix(lower);

    // 3. triu: Upper triangular extraction (k = 0)
    final upper = triu(a);
    print('\nUpper Triangular (triu, k=0):');
    _printMatrix(upper);

    // 4. Diagonal offsets (k = 1 and k = -1)
    final lowerK1 = tril(a, k: 1);
    print('\nLower Triangular with positive offset (tril, k=1):');
    _printMatrix(lowerK1);

    final upperKM1 = triu(a, k: -1);
    print('\nUpper Triangular with negative offset (triu, k=-1):');
    _printMatrix(upperKM1);

    // 5. Memory-efficient Recycling Buffer Reuse
    final recycler = NDArray<Float64>.zeros([3, 3], DType.float64);
    final recycledLower = tril(a, k: 0, out: recycler);
    print(
      '\nRecycled Output Buffer (identical check): ${identical(recycledLower, recycler) ? "PASS" : "FAIL"}',
    );
    _printMatrix(recycledLower);
  });
}

void _printMatrix(NDArray a) {
  final rows = a.shape[0];
  final cols = a.shape[1];
  for (var r = 0; r < rows; r++) {
    final rowStr = [];
    for (var c = 0; c < cols; c++) {
      final val = a.getCell([r, c]);
      rowStr.add((val as num).toStringAsFixed(1).padLeft(5));
    }
    print(' [ ${rowStr.join(', ')} ]');
  }
}
