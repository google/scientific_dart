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

/// An example of implementing the PageRank algorithm using the `ndarray` package.
///
/// This example demonstrates:
/// - Creating a transition probability matrix.
/// - Performing matrix-vector multiplication using [matmul].
/// - Using broadcasting and scalar operations for damping factor application.
/// - Normalizing vectors using [sum].
/// - Checking convergence using [allclose].
void main() {
  print('=== PageRank Algorithm Example ===\n');

  // Define the number of nodes in our web graph.
  const int n = 5;

  // Damping factor (probability of following a link).
  const double d = 0.85;

  // Convergence tolerance.
  const double tol = 1e-6;

  // Maximum number of iterations.
  const int maxIterations = 100;

  // 1. Create the transition probability matrix M (5x5).
  // M[i][j] is the probability of transitioning from node j to node i.
  // Columns must sum to 1 (column-stochastic).
  final m = NDArray<Float64>.fromList(
    ([
      0.0, 0.0, 1.0, 0.0, 0.0, // Row 0 (links to 0 from: 2)
      0.5, 0.0, 0.0, 0.0, 0.0, // Row 1 (links to 1 from: 0)
      0.5, 1.0, 0.0, 0.5, 1.0, // Row 2 (links to 2 from: 0, 1, 3, 4)
      0.0, 0.0, 0.0, 0.0, 0.0, // Row 3 (links to 3 from: none)
      0.0, 0.0, 0.0, 0.5, 0.0, // Row 4 (links to 4 from: 3)
    ]).map((e) => Float64(e)).toList(),
    [n, n],
    DType.float64,
  );

  print('Transition Matrix M:');
  _printMatrix(m);

  // Verify columns sum to 1.
  final colSums = sum<Float64>(m, axis: 0);
  print('\nVerify columns sum to 1: ${colSums.toList()}');
  colSums.dispose();

  // 2. Initialize PageRank vector v (uniform distribution).
  var v = NDArray<Float64>.fromList(
    List<Float64>.generate(n, (_) => Float64(1.0 / n)),
    [n],
    DType.float64,
  );

  print('\nInitial PageRank vector v: ${v.toList()}');

  // Teleportation constant: (1 - d) / N
  const double teleport = (1.0 - d) / n;

  // 3. Power Iteration
  var converged = false;
  var iteration = 0;

  for (iteration = 1; iteration <= maxIterations; iteration++) {
    final vNext = NDArray.scope(() {
      // v_next = d * (M * v) + (1 - d)/N

      // Matrix-vector multiplication
      final mv = matmul<Float64, Float64, Float64>(m, v);

      // Wrap scalar d as a 0D array for broadcasting
      final dArr = NDArray<Float64>.scalar(Float64(d), dtype: DType.float64);
      // Apply damping factor (scalar multiplication via broadcasting)
      final damped = multiply<Float64, Float64, Float64>(mv, dArr);

      // Wrap scalar teleport as a 0D array for broadcasting
      final teleportArr = NDArray<Float64>.scalar(
        Float64(teleport),
        dtype: DType.float64,
      );
      // Add teleportation (scalar addition via broadcasting)
      final rawNext = add<Float64, Float64, Float64>(damped, teleportArr);

      // Normalize using L1 norm (sum) to handle potential numerical drift
      final s = sum<Float64>(rawNext);
      final normalized = divide<Float64, Float64, Float64>(rawNext, s);

      // We must detach the result from the scope so it survives when the scope exits.
      return normalized.detachFromScope();
    });

    // Check convergence: allClose(v, vNext, rtol, atol)
    if (allClose(v, vNext, rtol: tol, atol: tol)) {
      converged = true;
      v.dispose();
      v = vNext;
      break;
    }

    // Dispose the old vector and update to the new one
    v.dispose();
    v = vNext;
  }

  if (converged) {
    print('\nConverged after $iteration iterations.');
  } else {
    print('\nDid not converge after $maxIterations iterations.');
  }

  print('\nFinal PageRank vector:');
  for (var i = 0; i < n; i++) {
    print('Node $i: ${v[[i]].toStringAsFixed(6)}');
  }

  final finalSum = sum<Float64>(v).scalar;
  print('\nSum of final PageRank vector: $finalSum');

  // Clean up manually managed arrays
  v.dispose();
  m.dispose();
}

/// Helper function to print a 2D matrix in a readable format.
void _printMatrix(NDArray matrix) {
  final rows = matrix.shape[0];
  final cols = matrix.shape[1];
  for (var i = 0; i < rows; i++) {
    final rowList = <Object?>[];
    for (var j = 0; j < cols; j++) {
      rowList.add(matrix[[i, j]]);
    }
    print(rowList);
  }
}
