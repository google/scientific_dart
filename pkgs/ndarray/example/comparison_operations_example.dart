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
  print('=== NDArray Element-Wise Comparisons & Mask Recycling Examples ===\n');

  runBasicComparisonsExample();
  runMaskRecyclingExample();
}

void runBasicComparisonsExample() {
  print('--- 1. Broadcasted Comparisons ---');
  final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
  final b = NDArray.fromList([10.0, 99.0, 30.0], [3], DType.float64);

  print('Array A: ${a.toList()}');
  print('Array B: ${b.toList()}');

  // equal
  final eq = equal(a, b);
  print('equal(A, B): ${eq.toList()}');

  // greater
  final gt = greater(b, a);
  print('greater(B, A): ${gt.toList()}');

  // lessEqual
  final lte = lessEqual(a, b);
  print('lessEqual(A, B): ${lte.toList()}\n');
}

void runMaskRecyclingExample() {
  print('--- 2. Allocation-Free Mask Recycling inside Loops ---');
  final dataset = NDArray.fromList(
    [0.5, 1.2, -0.8, 2.5, 0.1],
    [5],
    DType.float64,
  );
  final threshold = NDArray.fromList(
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [5],
    DType.float64,
  );

  print('Dataset: ${dataset.toList()}');
  print('Threshold bounds: ${threshold.toList()}');

  // Pre-allocate a boolean mask result buffer once!
  final recycledMask = NDArray<bool>.create([5], DType.boolean);

  print('\nIteratively comparing dataset against thresholds bounds...');
  const iterations = 5;
  for (var step = 1; step <= iterations; step++) {
    // Write the comparison result directly in-place into recycledMask!
    // No intermediate memory is allocated!
    greater(dataset, threshold, out: recycledMask);

    print('Step $step -> recycledMask data: ${recycledMask.toList()}');
  }

  print(
    '\n🏆 Mask buffer recycled successfully in-place with 0 memory allocations!',
  );
}
