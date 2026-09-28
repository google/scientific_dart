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
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  print('=== GpuArray Baseline Example ===\n');

  // 1. Basic GPU Array Creation & Arithmetic
  ResourceScope.scope(() {
    print('1. Creating GPU Arrays in VRAM:');
    final a = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
    final b = GpuArray.fromList(
      [10.0, 20.0, 30.0, 40.0],
      [2, 2],
      DType.float64,
    );

    print('a = ${a.toList()}');
    print('b = ${b.toList()}');

    // Chained elementwise ufuncs on GPU
    final c = (a * 2.0) + b;
    print('(a * 2.0) + b = ${c.toList()}\n');

    // 2. Matrix Multiplication on GPU
    print('2. Matrix Multiplication:');
    final m1 = GpuArray.fromList(
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
      [2, 2],
      DType.float64,
    );
    final m2 = GpuArray.fromList(
      [
        [5.0, 6.0],
        [7.0, 8.0],
      ],
      [2, 2],
      DType.float64,
    );

    final matmulResult = m1.matmul(m2);
    print('m1 @ m2 = ${matmulResult.toList()}\n');

    // 3. Reductions
    print('3. Reductions:');
    print('Sum total: ${matmulResult.sum().scalar}');
    print('Mean per column (axis 0): ${matmulResult.mean(axis: 0).toList()}');
    print('Max per row (axis 1): ${matmulResult.max(axis: 1).toList()}\n');
  });

  // 4. Seamless Interoperability with host NDArray
  print('4. Seamless Interop with Host NDArray:');
  ResourceScope.scope(() {
    // Start with a host NDArray
    final hostArr = NDArray.fromList([1.0, 4.0, 9.0, 16.0], [4], DType.float64);
    print('Host NDArray: ${hostArr.toList()}');

    // Upload to GPU via .toGpu() extension
    final gpuArr = hostArr.toGpu();
    print('Uploaded to GPU device: ${gpuArr.device.name}');

    // Execute compute kernel on GPU
    final gpuSqrt = gpuArr.sqrt();

    // Download back to host memory as an NDArray
    final resultND = gpuSqrt.toNDArray();
    print('Downloaded result NDArray: ${resultND.toList()}');
  });

  print('\n=== Example Finished Successfully ===');
}
