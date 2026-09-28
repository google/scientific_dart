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
  print(
    '=== NDArray expand_dims() and squeeze() Shape View Manipulation ===\n',
  );

  // 1. Create a 1D vector of shape [3]
  final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
    3,
  ], DType.float64);
  print('a: shape ${a.shape}, strides ${a.strides}, data ${a.toList()}');

  // 2. Expand dimensions at axis 0 -> shape [1, 3]
  final aExpand0 = expand_dims(a, 0);
  print(
    '\nexpand_dims(a, 0): shape ${aExpand0.shape}, strides ${aExpand0.strides}',
  );
  print('Is aExpand0 a zero-copy view? ${aExpand0.isView}'); // true

  // 3. Expand dimensions at axis 1 -> shape [3, 1]
  final aExpand1 = expand_dims(a, 1);
  print(
    'expand_dims(a, 1): shape ${aExpand1.shape}, strides ${aExpand1.strides}',
  );

  // 4. Squeeze dimensions
  // Create a 3D tensor of shape [1, 3, 1]
  final tensor = NDArray.fromList(Float64List.fromList([10.0, 20.0, 30.0]), [
    1,
    3,
    1,
  ], DType.float64);
  print('\nRaw tensor: shape ${tensor.shape}, strides ${tensor.strides}');

  // Squeeze all axes of size 1 -> shape [3]
  final squeezedAll = squeeze(tensor);
  print(
    'squeeze(tensor): shape ${squeezedAll.shape}, strides ${squeezedAll.strides}',
  );

  // Squeeze only axis 0 -> shape [3, 1]
  final squeezed0 = squeeze(tensor, axis: [0]);
  print(
    'squeeze(tensor, axis: [0]): shape ${squeezed0.shape}, strides ${squeezed0.strides}',
  );

  // Cleanup memory
  a.dispose();
  aExpand0.dispose();
  aExpand1.dispose();
  tensor.dispose();
  squeezedAll.dispose();
  squeezed0.dispose();
}
