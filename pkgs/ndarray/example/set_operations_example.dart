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
  print('=== unique ===');
  final a = NDArray<int>.fromList([1, 2, 2, 3, 1, 4], [6], DType.int32);
  final u = unique(a);
  print('Input: ${a.toList()}');
  print('Unique: ${u.toList()}'); // [1, 2, 3, 4]
  a.dispose();
  u.dispose();

  print('\n=== intersect1d ===');
  final ar1 = NDArray<int>.fromList([1, 3, 4, 3], [4], DType.int32);
  final ar2 = NDArray<int>.fromList([3, 1, 2, 1], [4], DType.int32);
  final intersection = intersect1d(ar1, ar2);
  print('Array 1: ${ar1.toList()}');
  print('Array 2: ${ar2.toList()}');
  print('Intersection: ${intersection.toList()}'); // [1, 3]
  ar1.dispose();
  ar2.dispose();
  intersection.dispose();

  print('\n=== setdiff1d ===');
  final sd1 = NDArray<int>.fromList([1, 2, 3, 2, 4], [5], DType.int32);
  final sd2 = NDArray<int>.fromList([2, 3, 5], [3], DType.int32);
  final diff = setdiff1d(sd1, sd2);
  print('Array 1: ${sd1.toList()}');
  print('Array 2: ${sd2.toList()}');
  print('Difference (1 - 2): ${diff.toList()}'); // [1, 4]
  sd1.dispose();
  sd2.dispose();
  diff.dispose();

  print('\n=== setxor1d ===');
  final sx1 = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
  final sx2 = NDArray<int>.fromList([2, 3, 4], [3], DType.int32);
  final xor = setxor1d(sx1, sx2);
  print('Array 1: ${sx1.toList()}');
  print('Array 2: ${sx2.toList()}');
  print('XOR: ${xor.toList()}'); // [1, 4]
  sx1.dispose();
  sx2.dispose();
  xor.dispose();

  print('\n=== union1d ===');
  final un1 = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
  final un2 = NDArray<int>.fromList([2, 3, 4, 5], [4], DType.int32);
  final union = union1d(un1, un2);
  print('Array 1: ${un1.toList()}');
  print('Array 2: ${un2.toList()}');
  print('Union: ${union.toList()}'); // [1, 2, 3, 4, 5]
  un1.dispose();
  un2.dispose();
  union.dispose();

  print('\n=== isin ===');
  final element = NDArray<int>.fromList([1, 2, 3, 4, 2, 1], [6], DType.int32);
  final testElements = NDArray<int>.fromList([2, 4], [2], DType.int32);
  final mask = isin(element, testElements);
  print('Element: ${element.toList()}');
  print('Test Elements: ${testElements.toList()}');
  print('Is In: ${mask.toList()}'); // [false, true, false, true, true, false]
  element.dispose();
  testElements.dispose();
  mask.dispose();
}
