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
  print(
    '=== NDArray Discrete Fourier Transform (FFT) Strides Correctness Examples ===\n',
  );

  runFFTStridedCorrectnessExample();
}

void runFFTStridedCorrectnessExample() {
  NDArray.scope(() {
    print('--- 1. Allocating Contiguous Signals Parent Array ---');
    // We create a 2D array of shape [2, 2]
    final parent = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0],
      [2, 2],
      DType.float64,
    );
    print('Parent array data:\n$parent');
    print(
      'Parent strides: ${parent.strides}, isContiguous: ${parent.isContiguous}',
    );

    print('\n--- 2. Generating Transposed Non-Contiguous View ---');
    // Transposing swaps axis 0 and axis 1 -> yields [[1, 3], [2, 4]]
    final transposed = parent.transposed;
    print('Transposed array data:\n$transposed');
    print(
      'Transposed strides: ${transposed.strides}, isContiguous: ${transposed.isContiguous}',
    );

    print('\n--- 3. Executing FFT on Transposed Signals View ---');
    // Automatically duplicates and processes transposed coordinates perfectly!
    final freqTransposed = fft(transposed);
    print('Transposed FFT results DType: ${freqTransposed.dtype}');
    print('Transposed FFT results flat data: ${freqTransposed.toList()}');

    print('\n--- 4. Comparing against Contiguous Equivalent Array ---');
    // Standard contiguous equivalent array [[1, 3], [2, 4]]
    final contiguous = NDArray.fromList(
      [1.0, 3.0, 2.0, 4.0],
      [2, 2],
      DType.float64,
    );
    final freqContig = fft(contiguous);
    print('Contiguous equivalent FFT flat data: ${freqContig.toList()}');

    // Check mathematical parity using allClose!
    final match = allClose(freqTransposed, freqContig, rtol: 1e-5, atol: 1e-5);
    print(
      '\n🏆 Numerical parity check between transposed and contiguous FFT: ${match ? "SUCCESS" : "FAILED"}',
    );
  });
}
