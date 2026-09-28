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
  print('=== NDArray Shaping & Meshes Examples ===\n');
  runAsStridedExamples();
  runMGridExamples();
  runOGridExamples();
}

void runAsStridedExamples() {
  print('--- 1. Low-level Striding View (asStrided) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
    print('Original 1D array: ${a.toList()}');

    // Create a 2x2 strided view pointing to the same memory
    final view = asStrided(a, shape: [2, 2], strides: [2, 1]);
    print('2x2 strided view: ${view.toList()}');

    // Mutating the view affects the original array
    view.setCell([0, 0], Int32(99));
    print('After modifying view[0, 0] to 99:');
    print('View: ${view.toList()}');
    print('Original: ${a.toList()}\n');
  });
}

void runMGridExamples() {
  print('--- 2. Dense Meshgrid (mgrid) ---');
  NDArray.scope(() {
    // Generate dense meshgrid of shape [2, 3, 2]
    final grid = mgrid([
      GridRange(0, 3), // 0 to 2 inclusive
      GridRange(0, 2), // 0 to 1 inclusive
    ]);

    print('Grid shape: ${grid.shape}');
    print('Grid coordinates: ${grid.toList()}');

    // Check individual coordinate arrays
    final x = grid.slice([Index(0)]);
    final y = grid.slice([Index(1)]);
    print('X coordinate grid:\n$x');
    print('Y coordinate grid:\n$y\n');
  });
}

void runOGridExamples() {
  print('--- 3. Open Meshgrid (ogrid) ---');
  NDArray.scope(() {
    // Generate open meshgrid of two arrays with exclusive step and inclusive complex step
    final grid = ogrid([
      GridRange(0, 3), // [0.0, 1.0, 2.0]
      GridRange.numpy(
        0,
        1,
        Complex(0, 3),
      ), // 3 points inclusive: [0.0, 0.5, 1.0]
    ]);

    print('X grid shape: ${grid[0].shape}');
    print('X grid: ${grid[0].toList()}');
    print('Y grid shape: ${grid[1].shape}');
    print('Y grid: ${grid[1].toList()}\n');
  });
}
