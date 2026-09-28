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
  print('=== NDArray nan_to_num() Dataset Sanitation Example ===\n');

  NDArray.scope(() {
    // 1. Initialize a Float64 vector containing NaN and Infinities
    final a = NDArray.fromList(
      Float64List.fromList([
        1.0,
        double.nan,
        double.infinity,
        -2.0,
        double.negativeInfinity,
      ]),
      [5],
      DType.float64,
    );
    print('Raw array a: ${a.toList()}');

    // 2. Default cleaning: NaN -> 0.0, inf -> max_float, -inf -> min_float
    final cleanDefault = nan_to_num(a);
    print('\nCleaned (default): ${cleanDefault.toList()}');

    // 3. Custom cleaning: NaN -> 99.0, posinf -> 500.0, neginf -> -500.0
    final cleanCustom = nan_to_num(a, nan: 99.0, posinf: 500.0, neginf: -500.0);
    print('Cleaned (custom): ${cleanCustom.toList()}');

    // 4. Blazingly fast in-place recycling to completely bypass allocations!
    print('\n--- Allocation-Free In-Place Recycler ---');
    nan_to_num(a, nan: 0.0, out: a); // mutate a in-place!
    print('Raw array a (after in-place clean): ${a.toList()}');
  });
}
