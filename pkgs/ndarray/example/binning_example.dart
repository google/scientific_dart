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
  print('=== NDArray Binning Examples ===\n');

  NDArray.scope(() {
    // 1. bincount example
    print('--- bincount ---');
    final a = NDArray<DTypeTag>.fromList(
      [0, 1, 1, 3, 2, 1, 7],
      [7],
      DType.int32,
    );
    final counts = bincount(a);
    print('bincount([0, 1, 1, 3, 2, 1, 7]): ${counts.toList()}');

    // 2. digitize example
    print('\n--- digitize ---');
    final x = NDArray<Float64>.fromList(
      [0.2, 6.4, 3.0, 1.6],
      [4],
      DType.float64,
    );
    final bins = NDArray<Float64>.fromList(
      [0.0, 1.0, 2.5, 4.0, 10.0],
      [5],
      DType.float64,
    );
    final inds = digitize(x, bins);
    print('digitize: ${inds.toList()}');

    // 3. histogram example
    print('\n--- histogram ---');
    final sample = NDArray<Float64>.fromList(
      [1.0, 2.0, 1.0],
      [3],
      DType.float64,
    );
    final (:hist, :binEdges) = histogram(sample, bins: 2, range: (0.0, 2.0));
    print('histogram counts: ${hist.toList()}');
    print('histogram bin edges: ${binEdges.toList()}');
  });
}
