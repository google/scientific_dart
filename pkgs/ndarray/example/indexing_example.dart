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
  print('=== NDArray Advanced Fancy Indexing & Syntax Examples ===\n');
  runStaticTypedAddressingExample();
  runPolymorphicNumpyParityExample();
}

void runStaticTypedAddressingExample() {
  print('--- 1. Explicit Statically Typed Addressing Methods ---');
  final mat = NDArray.fromList(Float64List.fromList([10.0, 20.0, 30.0, 40.0]), [
    2,
    2,
  ], DType.float64);

  // getCell and setCell are completely non-fancy, strongly typed, and predictable
  final cellVal = mat.getCell([0, 1]);
  print('getCell([0, 1]) -> expected 20.0: $cellVal');

  mat.setCell([1, 0], Float64(99.0));
  print(
    'After setCell([1, 0], 99.0) data -> expected [10, 20, 99, 40]: ${mat.toList()}',
  );

  // setByMask is explicit for boolean criteria
  final arr = NDArray.fromList(Float64List.fromList([-1.0, 5.0, -3.0, 8.0]), [
    4,
  ], DType.float64);
  final negativeMask = arr < 0.0; // returns an NDArray<int> mask

  // Clips all negative elements to 0.0 explicitly!
  arr.setByMaskScalar(negativeMask, Float64(0.0));
  print(
    'After setByMask clipping negatives -> expected [0, 5, 0, 8]: ${arr.toList()}',
  );

  // setIndices and setIndicesScalar are explicit for fancy index lists
  final bigVec = NDArray.fromList(Int32List.fromList([10, 20, 30, 40, 50]), [
    5,
  ], DType.int32);
  final targetIndices = NDArray.fromList([0, 4], [2], DType.int32);

  bigVec.setIndicesScalar(targetIndices, Int32(999));
  print(
    'After setIndicesScalar positions [0, 4] -> expected [999, 20, 30, 40, 999]: ${bigVec.toList()}\n',
  );
}

void runPolymorphicNumpyParityExample() {
  print('--- 2. NumPy-Equivalent Polymorphic Square-Bracket Overloads ---');
  final a = NDArray.arange(
    0.0,
    12.0,
    dtype: DType.int32,
  ).reshape([3, 4]); // 3 rows, 4 columns
  print('Matrix A (3x4 arange):');
  print('[0, 1, 2, 3]\n[4, 5, 6, 7]\n[8, 9, 10, 11]');

  // NumPy Parity A: arr[int] returns a direct sub-matrix row view!
  final rowView = a[1]; // extracts row 1
  print('a[1] (Row 1 view) -> expected [4, 5, 6, 7]: ${rowView.toList()}');

  // NumPy Parity B: arr[List<int>] triggers Fancy Indexing along the first axis!
  final fancyRows =
      a[[
        [0, 2],
      ]]; // extracts row 0 and row 2
  print(
    'a[[ [0, 2] ]] (Fancy row extraction) -> shape ${fancyRows.shape}, data: ${fancyRows.toList()}',
  );

  // NumPy Parity C: arr[arr > cond] triggers Boolean Mask criteria, flattening matches to 1D!
  final maskCondition = a > 7; // boolean matrix mask
  final filteredVec = a[maskCondition];
  print(
    'a[a > 7] (Boolean filtering vector) -> expected [8, 9, 10, 11]: ${filteredVec.toList()}',
  );

  // NumPy Parity D: Mutations via operator []= overloads!
  final data = NDArray.fromList(Float64List.fromList([-2.0, 10.0, -5.0, 4.0]), [
    4,
  ], DType.float64);

  // Boolean condition assignment clips elements in-place!
  data[data < 0.0] = 0.0;
  print(
    'Mutation data[data < 0.0] = 0.0 -> expected [0, 10, 0, 4]: ${data.toList()}',
  );

  // Fancy list assignment sets specific element indices in-place!
  data[[
        [1, 3],
      ]] =
      99.0;
  print(
    'Mutation data[[1, 3]] = 99.0 -> expected [0, 99, 0, 99]: ${data.toList()}',
  );
}
