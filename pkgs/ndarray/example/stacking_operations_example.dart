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
  print('=== NDArray New-Axis Stacking (stack) Examples ===\n');

  runCoordinateGridStackingExample();
  runMultidimensionalMatrixStackingExample();
  runStackVsConcatenateComparison();
}

void runCoordinateGridStackingExample() {
  print('--- 1. Stacking Coordinate Vectors into 2D Grid ---');
  final xCoords = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
  final yCoords = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);

  print('X Coordinates: ${xCoords.toList()}');
  print('Y Coordinates: ${yCoords.toList()}');

  // Stack them along axis 0 -> yields [[1, 2, 3], [10, 20, 30]] (shape [2, 3])
  final stackedAxis0 = stack([xCoords, yCoords], axis: 0);
  print(
    '\nStacked along axis 0 (new leading dimension, shape: ${stackedAxis0.shape}):',
  );
  print(stackedAxis0.toList());

  // Stack them along axis 1 -> yields [[1, 10], [2, 20], [3, 30]] (shape [3, 2])
  final stackedAxis1 = stack([xCoords, yCoords], axis: 1);
  print(
    '\nStacked along axis 1 (new trailing dimension, shape: ${stackedAxis1.shape}):',
  );
  print(stackedAxis1.toList());
}

void runMultidimensionalMatrixStackingExample() {
  print('\n--- 2. Stacking 2D Matrices into 3D Batches ---');
  final mat1 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  final mat2 = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);

  print('Matrix 1: ${mat1.toList()}');
  print('Matrix 2: ${mat2.toList()}');

  // Stack 2D matrices along axis 0 -> yields shape [2, 2, 2]
  final batch = stack([mat1, mat2], axis: 0);
  print('\nStacked 3D Batch along axis 0 (shape: ${batch.shape}):');
  print(batch.toList());

  // We can extract individual matrix items from the batch view!
  final item0 = batch.slice([Index(0), Slice.all(), Slice.all()]);
  print('Extracted Item 0 from Batch: ${item0.toList()}');
}

void runStackVsConcatenateComparison() {
  print('\n--- 3. Stacking vs Concatenation Comparison ---');
  final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
  final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);

  print('Array a: ${a.toList()}');
  print('Array b: ${b.toList()}');

  // concatenate() joins along an EXISTING axis -> stays a 1D vector of shape [4]
  final concatenated = concatenate([a, b], axis: 0);
  print(
    '\nConcatenated along existing axis 0 (stays 1D, shape: ${concatenated.shape}):',
  );
  print(concatenated.toList());

  // stack() joins along a BRAND NEW axis -> expands to a 2D matrix of shape [2, 2]
  final stacked = stack([a, b], axis: 0);
  print('Stacked along new axis 0 (expands to 2D, shape: ${stacked.shape}):');
  print(stacked.toList());
}
