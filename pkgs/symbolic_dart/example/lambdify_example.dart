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

import 'package:ndarray/ndarray.dart' hide sin, exp;
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  final x = Symbol('x');
  final y = Symbol('y');

  // Define symbolic wave equation z = sin(x) * exp(-0.1 * y)
  final expr = sin(x) * exp(Real(-0.1) * y);
  print('Symbolic formula: $expr');

  // Compile into a vectorized numerical lambda function
  final lambda = expr.lambdify([x, y]);

  // Create 2D coordinates array
  final xArr = NDArray.fromList(
    [0.0, 1.57079632679, 3.14159265359, 4.71238898038],
    [4],
    DType.float64,
  );
  final yArr = NDArray.fromList([0.0, 10.0, 20.0, 30.0], [4], DType.float64);

  // Evaluate across multi-dimensional array inputs
  final zArr = lambda.callArray([xArr, yArr]);
  print('Vectorized evaluation shape: ${zArr.shape}');

  final iter = NDIter(zArr);
  var i = 0;
  while (iter.moveNext()) {
    print('z[$i] = ${zArr.getCell(iter.coords)}');
    i++;
  }
}
