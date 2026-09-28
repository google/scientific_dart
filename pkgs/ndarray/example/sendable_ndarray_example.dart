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

import 'dart:isolate';

import 'package:ndarray/ndarray.dart';

void main() async {
  print('=== SendableNDArray Multi-Isolate Concurrency Examples ===\n');

  await runCopyModeExample();
  await runBorrowModeExample();
}

/// Demonstrates safe, isolated message-passing with [SendableNDArray.fromCopy].
Future<void> runCopyModeExample() async {
  print('--- 1. Copy Mode (Transferable Data Across Isolates) ---');

  await NDArray.scope(() async {
    final array = NDArray<Float64>.ones([4, 4], DType.float64);
    print('Original array on main isolate:\n$array');

    // Create a self-contained transferable copy
    final sendable = array.toSendable();

    // Compute reduction on a background isolate
    final totalSum = await Isolate.run(() {
      return NDArray.scope(() {
        // Materialize reconstructs a fresh, scope-registered NDArray
        final workerArray = sendable.materialize();
        return sum(workerArray).scalar;
      });
    });

    print('Sum calculated on background worker isolate: $totalSum\n');
  });
}

/// Demonstrates zero-copy shared C memory mutation with [SendableNDArray.unsafeBorrow].
Future<void> runBorrowModeExample() async {
  print('--- 2. Borrow Mode (Zero-Copy Shared Memory Mutation) ---');

  await NDArray.scope(() async {
    final array = NDArray<Float64>.zeros([6], DType.float64);
    print('Original array before worker mutation: ${array.toList()}');

    // Borrow the raw memory address
    final borrowed = array.toSendableBorrow();

    // Spawn a worker isolate to mutate a slice in-place
    await Isolate.run(() {
      NDArray.scope(() {
        final view = borrowed.materializeView();
        // Mutate in-place
        for (var i = 0; i < view.shape[0]; i++) {
          view[i] = ((i + 1) * 10.0);
        }
      });
    });

    print(
      'Array on main isolate after in-place worker mutation: ${array.toList()}\n',
    );
  });
}
