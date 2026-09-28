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
  print('=== NDArray isClose() and allClose() Tolerance Comparisons ===\n');

  NDArray.scope(() {
    // 1. Main floating-point tolerance verification
    final a = NDArray.fromList(Float64List.fromList([1.0, 1.00001, 2.0]), [
      3,
    ], DType.float64);
    final b = NDArray.fromList(Float64List.fromList([1.0, 1.00002, 2.0]), [
      3,
    ], DType.float64);

    print('a: ${a.toList()}');
    print('b: ${b.toList()}');

    // Default tolerances: rtol = 1e-05, atol = 1e-08
    final closeDefault = isClose(a, b);
    print(
      '\nisClose (default): ${closeDefault.toList()}',
    ); // [true, false, true]

    // Stretch tolerances: rtol = 1e-04
    final closeStretched = isClose(a, b, rtol: 1e-04);
    print(
      'isClose (rtol = 1e-04): ${closeStretched.toList()}',
    ); // [true, true, true]

    // 2. allClose check
    final allCloseDefault = allClose(a, b);
    print('\nallClose (default): $allCloseDefault'); // false

    final allCloseStretched = allClose(a, b, rtol: 1e-04);
    print('allClose (rtol = 1e-04): $allCloseStretched'); // true
  });
}
