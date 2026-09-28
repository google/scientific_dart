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
  print('===========================================================');
  print('      NumPy to package:ndarray Quickstart Tutorial         ');
  print('===========================================================\n');

  runModule1CreationAndScope();
  runModule2IndexingAndSlicing();
  runModule3MathAndUfuncs();
  runModule4ViewsVsCopies();
  runModule5StackingAndSplitting();
  runModule6ReductionsAndLinAlg();
}

void runModule1CreationAndScope() {
  print('--- Module 1: Array Creation & Scoped Memory ---');
  // Always wrap computations in NDArray.scope() so unmanaged C heap memory is freed!
  NDArray.scope(() {
    // 1. From List (equivalent to np.array(...))
    final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
    print(
      '1. np.array([[1, 2], [3, 4]]):\nShape: ${a.shape}, Rank: ${a.rank}, Size: ${a.size}',
    );
    print(a.toList());

    // 2. zeros, ones, and full
    final z = NDArray.zeros([2, 3], DType.float64);
    final o = NDArray.ones([2, 3], DType.float32);
    final f = NDArray.full([2, 2], 42.0, dtype: DType.float64);
    print('\n2. np.zeros((2,3)) list view: ${z.toList()}');
    print('   np.ones((2,3)) list view: ${o.toList()}');
    print('   np.full((2,2), 42) list view: ${f.toList()}');

    // 3. arange and linspace
    final ar = NDArray.arange(0.0, 10.0, step: 2.0, dtype: DType.float64);
    final lin = linspace(0.0, 1.0, 5, dtype: DType.float64);
    print('\n3. np.arange(0, 10, 2): ${ar.toList()}');
    print('   np.linspace(0, 1, 5): ${lin.toList()}');

    // 4. 0-D Scalar vs 2D Identity Matrix
    final eye = NDArray.eye(3, DType.float64);
    final scalar = NDArray.scalar(99.9, dtype: DType.float64);
    print('\n4. np.eye(3):\n$eye');
    print(
      '   np.array(99.9) 0-D scalar value: ${scalar.scalar} (shape ${scalar.shape})\n',
    );
  });
}

void runModule2IndexingAndSlicing() {
  print('--- Module 2: Indexing, Slicing & Masking ---');
  NDArray.scope(() {
    final mat = NDArray.arange(
      0.0,
      12.0,
      step: 1.0,
      dtype: DType.float64,
    ).reshape([3, 4]);
    print('Matrix (3x4):\n$mat');

    // 1. Explicit Statically Typed Access (Zero Overhead, No Polymorphism)
    final cellVal = mat.getCell([1, 2]);
    print('Explicit mat.getCell([1, 2]): $cellVal');
    mat.setCell([1, 2], Float64(888.0));
    print('After mat.setCell([1, 2], 888.0): cell is ${mat.getCell([1, 2])}');

    // 2. Polymorphic Overloaded Operator [] (NumPy parity)
    final row1 = mat[1]; // Sub-matrix view of Row 1
    print('Polymorphic mat[1] (row 1 view): ${row1.toList()}');

    // 3. Slicing rows 0..2 and columns 1..3
    final subMatrix = mat.slice([
      Slice(start: 0, stop: 2),
      Slice(start: 1, stop: 3),
    ]);
    print('Slice mat[0:2, 1:3]:\n$subMatrix');

    // 4. Fancy Indexing & Boolean Masking
    final fancyRows =
        mat[[
          [0, 2],
        ]]; // Extract rows 0 and 2
    print('Fancy row selection mat[[ [0, 2] ]]:\n$fancyRows');

    // Masking: clip all values > 10 to 10.0
    final mask = mat > Float64(10.0);
    mat.setByMaskScalar(mask, Float64(10.0));
    print('After clipping values > 10 to 10.0:\n$mat\n');
  });
}

void runModule3MathAndUfuncs() {
  print('--- Module 3: Arithmetic & Universal Functions (ufuncs) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
    final b = NDArray.fromList([2.0, 4.0, 5.0], [3], DType.float64);

    print('a: ${a.toList()}');
    print('b: ${b.toList()}');

    // Operators + - * / ~/ %
    print('a + b: ${(a + b).toList()}');
    print('a * b (Hadamard product): ${(a * b).toList()}');
    print('a / b (True float division): ${(a / b).toList()}');
    print(
      'a ~/ b (Floor integer division with upfront zero-check): ${(a ~/ b).toList()}',
    );
    print('a % b (Remainder): ${(a % b).toList()}');

    // Mathematical ufuncs
    final sines = sin(a);
    final sqrts = sqrt(a);
    print('np.sin(a): ${sines.toList()}');
    print('np.sqrt(a): ${sqrts.toList()}\n');
  });
}

void runModule4ViewsVsCopies() {
  print('--- Module 4: Views vs Deep Copies ---');
  NDArray.scope(() {
    final orig = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

    // 1. Zero-copy reshape view
    final reshaped = orig.reshape([4]);
    print('reshaped isView? ${reshaped.isView}');
    reshaped.setCell([0], Float64(999.0)); // Mutates orig!
    print(
      'After mutating reshaped view, orig cell [0, 0] is: ${orig.getCell([0, 0])}',
    );

    // 2. Transpose (zero-copy strided view)
    final transposed = orig.transposed;
    print(
      'transposed isView? ${transposed.isView}, strides: ${transposed.strides}',
    );

    // 3. Deep C-contiguous copy
    final deepCopy = orig.copy();
    deepCopy.setCell([0, 0], Float64(1.0)); // Completely decoupled!
    print(
      'After mutating deepCopy[0,0]=1.0, orig[0,0] remains: ${orig.getCell([0, 0])}\n',
    );
  });
}

void runModule5StackingAndSplitting() {
  print('--- Module 5: Stacking & Splitting ---');
  NDArray.scope(() {
    final v1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
    final v2 = NDArray.fromList([3.0, 4.0], [2], DType.float64);

    // concatenate along existing dimension (stays 1D shape [4])
    final concat = concatenate([v1, v2], axis: 0);
    print(
      'concatenate([v1, v2], axis: 0) -> shape ${concat.shape}: ${concat.toList()}',
    );

    // stack adds a brand new axis (expands to 2D matrix shape [2, 2])
    final stacked = stack([v1, v2], axis: 0);
    print('stack([v1, v2], axis: 0) -> shape ${stacked.shape}:\n$stacked');

    // split into 2 equal sections
    final splits = split(concat, 2);
    print(
      'split(concat, 2) section 0: ${splits[0].toList()}, section 1: ${splits[1].toList()}\n',
    );
  });
}

void runModule6ReductionsAndLinAlg() {
  print('--- Module 6: Statistical Reductions & OpenBLAS Linear Algebra ---');
  NDArray.scope(() {
    final mat = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );
    print('Matrix (2x3):\n$mat');

    // Total sum (0-D scalar array)
    final totSum = sum(mat);
    print('np.sum(mat) (0-D scalar): ${totSum.scalar}');

    // Sum along columns (axis 0, collapses rows) -> shape [3]
    final colSums = sum(mat, axis: 0);
    print('np.sum(mat, axis=0): ${colSums.toList()}');

    // Mean along rows preserving rank (axis 1, keepdims: true) -> shape [2, 1]
    final rowMeans = mean(mat, axis: 1, keepdims: true);
    print(
      'np.mean(mat, axis=1, keepdims=True) shape ${rowMeans.shape}:\n$rowMeans',
    );

    // OpenBLAS Matrix Multiplication (2x3 @ 3x2 -> 2x2)
    final A = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );
    final B = NDArray.fromList(
      [7.0, 8.0, 9.0, 1.0, 2.0, 3.0],
      [3, 2],
      DType.float64,
    );
    final C = matmul(A, B);
    print('\nOpenBLAS matmul(A, B):\n$C');
  });
}
