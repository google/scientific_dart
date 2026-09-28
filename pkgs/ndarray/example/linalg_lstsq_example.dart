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
  print('=== NDArray linalg.lstsq() Least-Squares Example ===\n');

  NDArray.scope(() {
    // We want to fit a straight line y = c + m * x to the following data points:
    // (1, 2), (2, 3.9), (3, 6.1)
    //
    // The system of equations is A * [c, m]^T = B:
    // [1, 1] * [c, m]^T = 2
    // [1, 2] * [c, m]^T = 3.9
    // [1, 3] * [c, m]^T = 6.1
    final a = NDArray.fromList(
      [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
      [3, 2],
      DType.float64,
    );
    final b = NDArray.fromList([2.0, 3.9, 6.1], [3], DType.float64);

    print('Matrix (A):');
    _printMatrix(a);

    print('\nRight-Hand Side (b):');
    print('  ${b.toList()}');

    final res = lstsq<Float64, Float64, Float64>(a, b);

    print('\nLeast-Squares Solution (x):');
    print('  Intercept (c): ${res.x.getCell([0]).toStringAsFixed(4)}');
    print('  Slope (m):     ${res.x.getCell([1]).toStringAsFixed(4)}');

    print('\nSums of Squared Residuals:');
    print('  ${res.residuals.toList()}');

    print('\nEffective Rank of A:');
    print('  ${res.rank}');

    print('\nSingular Values of A:');
    print('  ${res.s.toList()}');
  });
}

void _printMatrix(NDArray a) {
  final rows = a.shape[0];
  final cols = a.shape[1];
  for (var r = 0; r < rows; r++) {
    final rowStr = [];
    for (var c = 0; c < cols; c++) {
      final val = a.getCell([r, c]);
      rowStr.add((val as num).toStringAsFixed(4).padLeft(9));
    }
    print(' [ ${rowStr.join(', ')} ]');
  }
}
