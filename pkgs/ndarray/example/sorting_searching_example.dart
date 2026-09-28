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
  NDArray.scope(() {
    print('=== NDArray Sorting, Searching & Counting Examples ===\n');
    runNativeSortExample();
    runComplexSortExample();
    runArgsortExample();
    runWhereTernaryExample();
    runNonzeroAndCountingExample();
    runArgMinMaxExample();
    runSearchSortedExample();
    runFindIndexExample();
  });
}

void runNativeSortExample() {
  NDArray.scope(() {
    print('--- Native Zero-Copy Heap Sorting (Rows/Columns) ---');
    // Create a 2D matrix
    final mat = NDArray.fromList(
      Float64List.fromList([5.0, 1.0, 3.0, 2.0, 6.0, 4.0]),
      [2, 3],
      DType.float64,
    );
    print('Original matrix:\nRow 0: [5.0, 1.0, 3.0]\nRow 1: [2.0, 6.0, 4.0]');

    // Sort along the last axis (axis 1, columns inside rows)
    // Bypasses Dart, runs native libc qsort directly on the heap!
    final sortedRows = sort(mat, axis: 1);
    print('Sorted along rows (axis 1):');
    print('Row 0: ${sortedRows.toList().sublist(0, 3)}'); // [1.0, 3.0, 5.0]
    print('Row 1: ${sortedRows.toList().sublist(3, 6)}'); // [2.0, 4.0, 6.0]

    // Sort along axis 0 (columns across rows)
    // Employs the axis-swapping trick to leverage qsort natively!
    final sortedCols = sort(mat, axis: 0);
    print('Sorted along columns (axis 0):');
    print('Row 0: ${sortedCols.toList().sublist(0, 3)}'); // [2.0, 1.0, 3.0]
    print('Row 1: ${sortedCols.toList().sublist(3, 6)}'); // [5.0, 6.0, 4.0]
  });
}

void runComplexSortExample() {
  NDArray.scope(() {
    print('\n--- NumPy-Compliant Complex Number Lexicographical Sort ---');
    final a = NDArray<DTypeTag>.create([4], DType.complex128);
    // Lexicographical rule: sorted by real part, then by imaginary part if reals match!
    a[0] = Complex(2.0, 5.0);
    a[1] = Complex(1.0, 10.0);
    a[2] = Complex(
      2.0,
      1.0,
    ); // shares real=2.0 with a[0] but has lower imag=1.0
    a[3] = Complex(0.0, 0.0);
    print('Original complex array: ${a.toList()}');

    final b = sort(a);
    print('Sorted complex array (Real first, then Imag): ${b.toList()}');
    // Expected: [Complex(0,0), Complex(1,10), Complex(2,1), Complex(2,5)]
  });
}

void runArgsortExample() {
  NDArray.scope(() {
    print('\n--- Indirect Index Sorting (argsort) ---');
    final a = NDArray.fromList(Float64List.fromList([50.0, 10.0, 40.0, 20.0]), [
      4,
    ], DType.float64);
    print('Array a: ${a.toList()}');

    // argsort returns the indices that would sort the array
    final indices = argsort(a);
    print(
      'Argsort indices: ${indices.toList()}',
    ); // [1, 3, 2, 0] -> element 10.0 is smallest, then 20.0, etc.

    // We can use take to reconstruct the sorted array via these indices!
    final sorted = a.take(indices.toList().cast<int>());
    print(
      'Reconstructed sorted array via take: ${sorted.toList()}',
    ); // [10.0, 20.0, 40.0, 50.0]
  });
}

void runWhereTernaryExample() {
  NDArray.scope(() {
    print('\n--- Ternary where Select (Fast SIMD Pipeline) ---');
    final cond = NDArray.fromList(
      [true, false, false, true],
      [4],
      DType.boolean,
    );
    final x = NDArray.fromList(Float32List.fromList([10.0, 20.0, 30.0, 40.0]), [
      4,
    ], DType.float32);
    final y = NDArray.fromList(Float32List.fromList([-1.0, -2.0, -3.0, -4.0]), [
      4,
    ], DType.float32);

    // Trigger hardware SIMD blending if supported by native implementation
    final result = where(cond, x, y);
    print('Condition: ${cond.toList()}');
    print('x: ${x.toList()}');
    print('y: ${y.toList()}');
    print(
      'where(cond, x, y) SIMD result: ${result.toList()}',
    ); // [10.0, -2.0, -3.0, 40.0]
  });
}

void runNonzeroAndCountingExample() {
  NDArray.scope(() {
    print('\n--- Non-zero Discovery (nonzero) & count_nonzero ---');
    final mat = NDArray.fromList(Int32List.fromList([0, 7, 0, 3, 0, 5]), [
      2,
      3,
    ], DType.int32);
    print('Matrix:\n[[0, 7, 0],\n [3, 0, 5]]');

    final nzCount = count_nonzero(mat).scalar;
    print('Total non-zero count: $nzCount'); // 3

    // nonzero returns coordinate arrays for each axis
    final nzCoords = nonzero(mat);
    print('nonzero returned ${nzCoords.length} coordinate arrays:');
    print('Axis 0 (row indices): ${nzCoords[0].toList()}'); // [0, 1, 1]
    print('Axis 1 (col indices): ${nzCoords[1].toList()}'); // [1, 0, 2]
    // The coordinates are: (0,1)=7, (1,0)=3, (1,2)=5!
  });
}

void runArgMinMaxExample() {
  NDArray.scope(() {
    print('\n--- Extremes Discovery (argmax / argmin) ---');
    final mat = NDArray.fromList(
      Float64List.fromList([10.0, 30.0, 20.0, 40.0, 5.0, 60.0]),
      [2, 3],
      DType.float64,
    );

    // Global flat extreme (axis = null)
    final globalMaxIdx = argmax(
      mat,
    ).scalar; // flattens to 6 elements, max is 60.0 at index 5
    final globalMinIdx = argmin(mat).scalar; // min is 5.0 at index 4
    print('Global flat argmax index: $globalMaxIdx'); // 5
    print('Global flat argmin index: $globalMinIdx'); // 4

    // Axis reduction argmax
    final axis0Max = argmax(mat, axis: 0); // find max row index for each column
    print(
      'argmax along axis 0 (columns max): ${axis0Max.toList()}',
    ); // [1, 0, 1] -> col0 max row is 1(40.0), col1 row 0(30.0), col2 row 1(60.0)
  });
}

void runSearchSortedExample() {
  NDArray.scope(() {
    print('\n--- Binary Search Insertion (searchsorted) ---');
    final a = NDArray.fromList(
      [10.0, 20.0, 30.0, 40.0, 50.0],
      [5],
      DType.float64,
    );
    final v = NDArray.fromList([15.0, 30.0, 5.0, 55.0], [4], DType.float64);
    print('Sorted array a: ${a.toList()}');
    print('Search values v: ${v.toList()}');

    // Left side: returns the first suitable index where element should be inserted
    final idxL = searchsorted(a, v, side: SearchSide.left);
    print('searchsorted(left): ${idxL.toList()}'); // [1, 2, 0, 5]

    // Right side: returns the last suitable index
    final idxR = searchsorted(a, v, side: SearchSide.right);
    print('searchsorted(right): ${idxR.toList()}'); // [1, 3, 0, 5]
  });
}

void runFindIndexExample() {
  print('\n--- Coordinates Discovery (findIndex) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([1.0, 2.0, 3.0, 2.0, 1.0], [5], DType.float64);
    print('Array a: ${a.toList()}');
    final coords = findIndex(a, CompareOp.equal, 2.0);
    print('findIndex(a, ==, 2.0) coordinates: $coords'); // [1]

    final a2d = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );
    print('2D Array:\n[[1.0, 2.0, 3.0],\n [4.0, 5.0, 6.0]]');
    final coords2d = findIndex(a2d, CompareOp.greater, 4.0);
    print('findIndex(a2d, >, 4.0) coordinates: $coords2d'); // [1, 1] (for 5.0)
  });
}
