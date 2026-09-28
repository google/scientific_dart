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
  NDArray.scope(() {
    print('--- NDArray Creation ---');
    final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
      2,
      2,
    ], DType.float64);
    print(
      'Array A:\nShape: ${a.shape}\nStrides: ${a.strides}\nData: ${a.toList()}',
    );

    print('\n--- Broadcasting Addition ---');
    final b = NDArray.fromList(Float64List.fromList([10, 20]), [
      2,
      1,
    ], DType.float64);
    final c = NDArray.fromList(Float64List.fromList([1, 2, 3]), [
      1,
      3,
    ], DType.float64);
    print('Array B shape: ${b.shape}');
    print('Array C shape: ${c.shape}');

    final d = add(b, c);
    print('Result B + C shape: ${d.shape}');
    print('Result B + C data: ${d.toList()}');

    print('\n--- Matrix Multiplication (OpenBLAS) ---');
    final m1 = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
      2,
      2,
    ], DType.float64);
    final m2 = NDArray.fromList(Float64List.fromList([5, 6, 7, 8]), [
      2,
      2,
    ], DType.float64);

    final m3 = matmul(m1, m2);
    print('Result m1 * m2 shape: ${m3.shape}');
    print('Result m1 * m2 data: ${m3.toList()}');
  });
}
