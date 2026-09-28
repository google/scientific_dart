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

import 'dart:math' as math;

import 'package:ndarray/ndarray.dart';

void main() {
  print('=== Root Finding and Optimization Example ===\n');

  NDArray.scope(() {
    // 1. Finding roots of 1D scalar functions using root_scalar (Brentq method)
    print('1. Root Finding with Brentq (Interval bracketing):');
    final brentResult = root_scalar(
      (x) => x * x - 4.0,
      method: RootMethod.brentq,
      bracketA: 0.0,
      bracketB: 3.0,
    );
    print('   Root of x^2 - 4 on [0, 3]: ${brentResult.root}');
    print(
      '   Converged: ${brentResult.converged}, Iterations: ${brentResult.iterations}\n',
    );

    // 2. Finding roots with Newton-Raphson (with analytical derivative)
    print('2. Root Finding with Newton-Raphson:');
    final newtonResult = root_scalar(
      (x) => math.cos(x) - x,
      method: RootMethod.newton,
      x0: 0.5,
      fprime: (x) => -math.sin(x) - 1.0,
    );
    print('   Root of cos(x) - x = 0: ${newtonResult.root}');
    print(
      '   Converged: ${newtonResult.converged}, Iterations: ${newtonResult.iterations}\n',
    );

    // 3. Multivariate Minimization using Nelder-Mead simplex
    print('3. Multivariate Minimization with Nelder-Mead:');
    final x0 = NDArray<Float64>.fromList([0.0, 0.0], [2], DType.float64);
    final minResult = minimize(
      (x) {
        final x1 = x.getCell([0]);
        final x2 = x.getCell([1]);
        return (x1 - 3.0) * (x1 - 3.0) + (x2 + 2.0) * (x2 + 2.0);
      },
      x0,
      method: MinimizeMethod.nelderMead,
    );
    print(
      '   Minimum found at: [${minResult.x.getCell([0])}, ${minResult.x.getCell([1])}]',
    );
    print('   Function value at min: ${minResult.fun}');
    print('   Success: ${minResult.success}, Iterations: ${minResult.nit}');
  });
}
