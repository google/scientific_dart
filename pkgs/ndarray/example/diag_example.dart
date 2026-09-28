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

import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray diag() Diagonal Matrix & Vector Example ===\n');

  NDArray.scope(() {
    // 1. Create a 2D matrix [3, 3]
    final mat = NDArray.fromList(
      Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
      [3, 3],
      DType.float64,
    );
    print(
      'Matrix mat:\n${mat.toList().sublist(0, 3)}\n${mat.toList().sublist(3, 6)}\n${mat.toList().sublist(6, 9)}',
    );

    // 2. Extract the main diagonal (k = 0) as a zero-copy 1D view!
    final mainDiag = diag(mat);
    print('\nExtract k = 0 diagonal: ${mainDiag.toList()}'); // [1.0, 5.0, 9.0]

    // 3. Extract upper diagonal (k = 1)
    final upperDiag = diag(mat, k: 1);
    print('Extract k = 1 diagonal: ${upperDiag.toList()}'); // [2.0, 6.0]

    // 4. Extract lower diagonal (k = -1)
    final lowerDiag = diag(mat, k: -1);
    print('Extract k = -1 diagonal: ${lowerDiag.toList()}'); // [4.0, 8.0]

    // 5. Construct a 2D diagonal matrix from a 1D vector
    final vec = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
      2,
    ], DType.float64);
    print('\nVector vec: ${vec.toList()}');

    final diagMat = diag(vec);
    print(
      'Diagonal Matrix from vec (k = 0):\n${diagMat.toList().sublist(0, 2)}\n${diagMat.toList().sublist(2, 4)}',
    );

    final upperDiagMat = diag(vec, k: 1);
    print(
      'Diagonal Matrix from vec (k = 1):\n${upperDiagMat.toList().sublist(0, 3)}\n${upperDiagMat.toList().sublist(3, 6)}\n${upperDiagMat.toList().sublist(6, 9)}',
    );
  });
}
