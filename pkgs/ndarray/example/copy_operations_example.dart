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
  print('=== NDArray Deep Copy (copy) ufunc Examples ===\n');

  runContiguousCopyExample();
  runStridedViewCopyExample();
}

void runContiguousCopyExample() {
  print('--- 1. Deep Copying a Contiguous Array ---');
  // Allocate standard 1D array
  final parent = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
  print('Parent array: ${parent.toList()}');

  // Perform deep copy using top-level copy() ufunc (equivalent to np.copy(a))
  final duplicate = copy(parent);
  print('Copied duplicate: ${duplicate.toList()}');

  // Let\'s verify memory decoupling (modifying copy does not affect parent!)
  print('\nModifying index 0 of duplicate to 99.0...');
  duplicate[0] = Float64(99.0);

  print('Parent array index 0: ${parent[0]}');
  print('Duplicate array index 0: ${duplicate[0]}');
  print('🏆 Memory blocks are successfully decoupled!');
}

void runStridedViewCopyExample() {
  print('\n--- 2. Deep Copying a Strided Transposed Array View ---');
  // Allocate 2D array [[1, 2], [3, 4]]
  final parent = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  print('Parent 2D array data:\n$parent');

  // Swapping axes creates a strided, non-contiguous view
  final transposedView = parent.transposed;
  print('Transposed view data:\n$transposedView');
  print('Transposed view isContiguous: ${transposedView.isContiguous}');

  // copy() automatically detects strided view, and duplicates a contiguous equivalent copy!
  final duplicate = copy(transposedView);
  print('\nCopied duplicate flat data: ${duplicate.toList()}');
  print('Copied duplicate isContiguous: ${duplicate.isContiguous}');

  // Decoupled memory verification
  duplicate[0] = Float64(99.0);
  print('\nModifying copy data[0] to 99.0...');
  print('Original parent data[0] (still 1.0): ${parent[0]}');
  print('🏆 Strided coordinates recursively deep copied successfully!');
}
