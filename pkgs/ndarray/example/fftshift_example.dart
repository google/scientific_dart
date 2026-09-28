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
  print('=== NDArray fftshift and ifftshift Spectrum Shifting Examples ===\n');

  NDArray.scope(() {
    // 1D Example (Odd Length)
    print('--- 1D Spectrum Shifting (Odd Length N=5) ---');
    final signal1D = NDArray.fromList(
      [0.0, 1.0, 2.0, 3.0, 4.0],
      [5],
      DType.float64,
    );
    print('Original 1D Array:   ${signal1D.toList()}');

    final shifted1D = fftshift(signal1D);
    print('After fftshift:      ${shifted1D.toList()}');

    final restored1D = ifftshift(shifted1D);
    print('After ifftshift:     ${restored1D.toList()}\n');

    // 1D Example (Even Length)
    print('--- 1D Spectrum Shifting (Even Length N=6) ---');
    final signalEven = NDArray.fromList(
      [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
      [6],
      DType.float64,
    );
    print('Original 1D Array:   ${signalEven.toList()}');

    final shiftedEven = fftshift(signalEven);
    print('After fftshift:      ${shiftedEven.toList()}');

    final restoredEven = ifftshift(shiftedEven);
    print('After ifftshift:     ${restoredEven.toList()}\n');

    // 2D Grid Example
    print('--- 2D Grid Spectrum Shifting (2D Shape: [2, 3]) ---');
    final grid = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );
    print('Original 2D Grid:');
    _print2DGrid(grid);

    final shifted2D = fftshift(grid);
    print('\nAfter fftshift (both axes):');
    _print2DGrid(shifted2D);

    final shifted2DAxis0 = fftshift(grid, axes: 0);
    print('\nAfter fftshift (axis 0 only):');
    _print2DGrid(shifted2DAxis0);

    final restored2D = ifftshift(shifted2D);
    print('\nAfter ifftshift (both axes, restored):');
    _print2DGrid(restored2D);
  });
}

void _print2DGrid(NDArray a) {
  final list = a.toList();
  final cols = a.shape[1];
  for (var r = 0; r < a.shape[0]; r++) {
    final row = list.sublist(r * cols, (r + 1) * cols);
    print('  $row');
  }
}
