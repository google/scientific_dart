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
  print('=== NDArray Vectorized Bitwise Operations & Recycling Examples ===\n');

  runBasicBitwiseExample();
  runRecyclingBitwiseExample();
}

void runBasicBitwiseExample() {
  print('--- 1. Basic Vectorized Bitwise Operations ---');
  final a = NDArray.fromList([5, 12, 3], [3], DType.int32);
  final b = NDArray.fromList([3, 4, 5], [3], DType.int32);

  print('Array A: ${a.toList()}');
  print('Array B: ${b.toList()}');

  // bitwise_and: 5 & 3 = 1, 12 & 4 = 4, 3 & 5 = 1
  final andRes = bitwise_and(a, b);
  print('bitwise_and(A, B): ${andRes.toList()}');

  // bitwise_or: 5 | 3 = 7, 12 | 4 = 12, 3 | 5 = 7
  final orRes = bitwise_or(a, b);
  print('bitwise_or(A, B): ${orRes.toList()}');

  // bitwise_xor: 5 ^ 3 = 6, 12 ^ 4 = 8, 3 ^ 5 = 6
  final xorRes = bitwise_xor(a, b);
  print('bitwise_xor(A, B): ${xorRes.toList()}');

  // left_shift: 5 << 1 = 10, 12 << 2 = 48
  final shiftAmount = NDArray.fromList([1, 2, 1], [3], DType.int32);
  final leftRes = left_shift(a, shiftAmount);
  print('left_shift(A, ShiftAmount): ${leftRes.toList()}');

  // invert (bitwise NOT)
  final invRes = invert(a);
  print('invert(A): ${invRes.toList()}\n');
}

void runRecyclingBitwiseExample() {
  print('--- 2. Allocation-Free Bitwise Recycling inside Loops ---');
  final a = NDArray.fromList([1, 2, 4, 8, 16], [5], DType.int64);
  final shift = NDArray.fromList([1, 1, 1, 1, 1], [5], DType.int64);

  print('Initial Array A: ${a.toList()}');

  // Pre-allocate a buffer once for the left shift result
  final recycledBuffer = NDArray<Int64>.create([5], DType.int64);

  print(
    '\nIteratively left-shifting array in-place with 0 memory allocations...',
  );
  const iterations = 3;
  for (var step = 1; step <= iterations; step++) {
    // Recycles the pre-allocated buffer to write the left shift results!
    left_shift(a, shift, out: recycledBuffer);
    print('Step $step -> left_shift result: ${recycledBuffer.toList()}');

    // Prepare next input by copying recycled buffer contents back
    for (var i = 0; i < a.size; i++) {
      a[i] = recycledBuffer[i];
    }
  }

  recycledBuffer.dispose();
  a.dispose();
  shift.dispose();
  print(
    '\n🏆 Bitwise operation recycled successfully in-place with 0 memory allocations!',
  );
}
