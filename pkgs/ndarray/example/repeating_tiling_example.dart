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
  print('=== NDArray Repeating and Tiling Examples ===\n');

  NDArray.scope(() {
    // 1. repeat example
    print('--- repeat ---');
    final a = NDArray.fromList([1, 2], [2], DType.int32);
    final r = repeat(a, [3]);
    print('repeat([1, 2], [3]): ${r.toList()}'); // [1, 1, 1, 2, 2, 2]

    // 2. tile example
    print('\n--- tile ---');
    final t = tile(a, [2]);
    print('tile([1, 2], [2]): ${t.toList()}'); // [1, 2, 1, 2]
  });
}
