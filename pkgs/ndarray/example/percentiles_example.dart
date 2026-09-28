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
  // We wrap everything in NDArray.scope to ensure native memory is freed.
  NDArray.scope(() {
    print('--- Median Examples ---');
    final a = NDArray.fromList([3.0, 1.0, 2.0, 4.0], [4], DType.float64);
    print('Array a: ${a.toList()}');
    print('Median of a (even size): ${median(a).scalar}'); // Expected: 2.5

    final b = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
    print('Array b: ${b.toList()}');
    print('Median of b (odd size): ${median(b).scalar}'); // Expected: 2.0

    // 2D Matrix
    // [[1, 5, 3],
    //  [4, 2, 6]]
    final mat = NDArray.fromList(
      [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
      [2, 3],
      DType.float64,
    );
    print('\n2D Matrix mat:\n$mat');

    final medAxis0 = median(mat, axis: 0);
    print(
      'Median along axis 0 (columns): ${medAxis0.toList()}',
    ); // Expected: [2.5, 3.5, 4.5]

    final medAxis1 = median(mat, axis: 1);
    print(
      'Median along axis 1 (rows): ${medAxis1.toList()}',
    ); // Expected: [3.0, 4.0]

    print('\n--- Percentile Examples ---');
    final data = NDArray.fromList(
      [15.0, 20.0, 35.0, 40.0, 50.0],
      [5],
      DType.float64,
    );
    print('Data: ${data.toList()}');

    // 40th percentile.
    // Indices: 0:15, 1:20, 2:35, 3:40, 4:50.
    // Target index: 4 * 0.4 = 1.6.
    // Interpolation: 20 + 0.6 * (35 - 20) = 29.0
    final p40 = percentile(data, 40.0);
    print('40th percentile: ${p40.scalar}'); // Expected: 29.0

    final p75 = percentile(data, 75.0);
    print('75th percentile: ${p75.scalar}'); // Expected: 40.0 (index 3)

    print('\n--- Quantile Examples ---');
    // Quantile is same as percentile but q is in [0, 1]
    final q04 = quantile(data, 0.4);
    print('0.4 quantile: ${q04.scalar}'); // Expected: 29.0
  });
}
