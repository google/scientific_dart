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
  print('=== NDArray sinh(), cosh(), and tanh() Hyperbolic Examples ===\n');

  NDArray.scope(() {
    // 1. Hyperbolic tangent activation loop
    final a = NDArray.fromList([-2.0, -1.0, 0.0, 1.0, 2.0], [5], DType.float64);
    print('Input activation values:');
    print(' [ ${a.toList().join(", ")} ]');

    final tanhAct = tanh(a);
    print('\nTanh Activation results (tanh):');
    print(
      ' [ ${tanhAct.toList().map((e) => e.toStringAsFixed(4)).join(", ")} ]',
    );

    final sinhVal = sinh(a);
    print('\nHyperbolic sine results (sinh):');
    print(
      ' [ ${sinhVal.toList().map((e) => e.toStringAsFixed(4)).join(", ")} ]',
    );

    final coshVal = cosh(a);
    print('\nHyperbolic cosine results (cosh):');
    print(
      ' [ ${coshVal.toList().map((e) => e.toStringAsFixed(4)).join(", ")} ]',
    );

    // 2. Inverse Hyperbolic sweeps
    final asinhVal = asinh(sinhVal);
    print('\nReconstructed values via inverse (asinh(sinh(x))):');
    print(
      ' [ ${asinhVal.toList().map((e) => e.toStringAsFixed(4)).join(", ")} ]',
    );
  });
}
