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
import 'dart:math' as math;

import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Transcendental Mathematical Ufuncs Example ===\n');

  NDArray.scope(() {
    // 1. Initialize a 1D vector array
    final x = NDArray.fromList(
      Float64List.fromList([0.0, math.pi / 2, math.pi]),
      [3],
      DType.float64,
    );
    print('x: ${x.toList()}');

    // 2. Compute sine and cosine
    final sineVal = sin(x);
    final cosineVal = cos(x);
    print('sin(x): ${sineVal.toList()}');
    print('cos(x): ${cosineVal.toList()}');

    // 3. Compute element-wise exponential and natural logarithm
    final expVal = exp(x);
    final logVal = log(expVal); // log(exp(x)) = x
    print('exp(x): ${expVal.toList()}');
    print('log(exp(x)): ${logVal.toList()}');

    // 4. Blazingly Fast Allocation-Free Recycling using named {out} parameter
    final outRecycler = NDArray<Float64>.zeros([3], DType.float64);
    print('\n--- Allocation-Free Out Recycling ---');
    print('outRecycler (before): ${outRecycler.toList()}');

    // Pass the out recycler to completely bypass transient array heap allocations!
    sin(x, out: outRecycler);
    print('outRecycler (after sin): ${outRecycler.toList()}');
  });
}
