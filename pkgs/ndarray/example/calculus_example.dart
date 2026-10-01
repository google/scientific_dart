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
  print('=== NDArray Calculus Examples ===\n');

  NDArray.scope(() {
    // 1. trapz example
    print('--- trapz ---');
    final y = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
    final integral = trapz(y, spacing: Spacing.step(1.0));
    print('trapz result: ${integral.scalar}'); // 4.5

    // 2. gradient example
    print('\n--- gradient ---');
    final f = NDArray.fromList([1.0, 2.0, 4.0, 7.0], [4], DType.float64);
    final grad = gradient(f, spacing: Spacing.step(1.0));
    print('gradient result: ${grad.toList()}'); // [1.0, 1.5, 2.5, 3.0]

    // 3. gradientArray example
    print('\n--- gradientArray ---');
    final f2D = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [2, 2], DType.float64);
    final grads = gradientArray(f2D, spacing: Spacing.step(1.0));
    print('gradientArray axis 0: ${grads[0].toList()}');
    print('gradientArray axis 1: ${grads[1].toList()}');
  });
}
