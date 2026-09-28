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
  NDArray.scope(() {
    // 1. Kronecker Product
    final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
    final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
    final k = kron(a, b);
    print('Kronecker Product: ${k.toList()}'); // [3.0, 4.0, 6.0, 8.0]

    // 2. Vector Outer Product
    final u = NDArray.fromList([1, 2], [2], DType.int32);
    final v = NDArray.fromList([3, 4], [2], DType.int32);
    final o = outer(u, v);
    print('Outer Product Matrix:\n${o.toList()}'); // [3, 4, 6, 8]

    // 3. Vector Cross Product (3D)
    final v1 = NDArray.fromList([1.0, 0.0, 0.0], [3], DType.float64);
    final v2 = NDArray.fromList([0.0, 1.0, 0.0], [3], DType.float64);
    final c = cross(v1, v2);
    print('Cross Product Vector: ${c.toList()}'); // [0.0, 0.0, 1.0]

    // 4. Vector and Matrix Norms
    final x = NDArray.fromList([1.0, -2.0, 3.0, -4.0], [4], DType.float64);
    final l1 = norm(x, ord: 1);
    print('Vector L1 Norm: ${l1.scalar}'); // 10.0

    final l2 = norm(x, ord: 2);
    print('Vector L2 Norm: ${l2.scalar}'); // sqrt(30)

    final m = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
    final fro = norm(m, ord: NormKind.frobenius);
    print('Matrix Frobenius Norm: ${fro.scalar}'); // sqrt(30)
  });
}
