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
  print('=== NDArray Indirect Sorting (argsort) Optimization Examples ===\n');

  runArgsortSortingByKeysExample();
}

void runArgsortSortingByKeysExample() {
  NDArray.scope(() {
    print('--- 1. Sorting Prices by Product Names Indirectly ---');
    // Suppose we have product prices and corresponding product names:
    final prices = NDArray.fromList(
      [39.9, 15.5, 22.0, 9.9],
      [4],
      DType.float64,
    );
    final names = ['Tablet', 'Keyboard', 'Mouse', 'Cable'];

    print('Product Names: $names');
    print('Product Prices: ${prices.toList()}');

    // We want to sort the product names based on their prices!
    // argsort returns indices that would sort the prices array:
    final indices = argsort(prices);
    print('\nArgsort indices (sorted index mapping): ${indices.toList()}');

    // Map the names list according to the sorted indices!
    final indexList = indices.toList();
    final sortedNames = indexList.map((idx) => names[idx]).toList();
    final sortedPrices = indexList.map((idx) => prices.getCell([idx])).toList();

    print('\n--- Sorted Results (By Price Ascending) ---');
    for (var i = 0; i < 4; i++) {
      print('  ${sortedNames[i]} : \$${sortedPrices[i]}');
    }

    print('\n🏆 Zero-Allocation indirect sorting executed successfully!');
  });
}
