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

import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('NDArray <-> GpuArray Interoperability', () {
    test('NDArray.toGpu() transfers data seamlessly to GpuArray', () {
      ResourceScope.scope(() {
        final hostND = nd.NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          nd.DType.float64,
        );

        // Convert to GpuArray using extension method
        final gpuArr = hostND.toGpu();
        expect(gpuArr.shape, equals([2, 2]));
        expect(gpuArr.dtype, equals(DType.float64));
        expect(
          gpuArr.toNestedList(),
          equals([
            [10.0, 20.0],
            [30.0, 40.0],
          ]),
        );
      });
    });

    test('GpuArray.toNDArray() transfers data back to host memory', () {
      ResourceScope.scope(() {
        final gpuArr = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float32,
        );

        final hostND = gpuArr.toNDArray();
        expect(hostND.shape, equals([2, 3]));
        expect(hostND.dtype, equals(nd.DType.float32));
        expect(hostND.toList(), equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
      });
    });

    test('Complete roundtrip workflow with computation in VRAM', () {
      ResourceScope.scope(() {
        // 1. Host inputs
        final aND = nd.NDArray.fromList([1.0, 2.0, 3.0], [3], nd.DType.float64);
        final bND = nd.NDArray.fromList(
          [10.0, 20.0, 30.0],
          [3],
          nd.DType.float64,
        );

        // 2. Upload to GPU
        final aGpu = aND.toGpu();
        final bGpu = bND.toGpu();

        // 3. Compute entirely on GPU
        final cGpu = aGpu + bGpu;
        final dGpu = cGpu * 2.0;

        // 4. Download back to NDArray
        final resultND = dGpu.toNDArray();

        // 5. Verify numerical results on host
        expect(resultND.toList(), equals([22.0, 44.0, 66.0]));
      });
    });

    test('Non-contiguous NDArray is handled cleanly during toGpu()', () {
      ResourceScope.scope(() {
        final hostND = nd.NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          nd.DType.float64,
        );
        // Create transposed view (non-contiguous)
        final transposedND = hostND.transpose();
        expect(transposedND.isContiguous, isFalse);

        final gpuArr = transposedND.toGpu();
        expect(
          gpuArr.toNestedList(),
          equals([
            [1.0, 3.0],
            [2.0, 4.0],
          ]),
        );
      });
    });
  });
}
