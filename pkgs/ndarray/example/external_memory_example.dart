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

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    print(
      '=== NDArray User-Allocated Memory (External Pointer) Examples ===\n',
    );

    runExternallyManagedExample();
    runCustomFinalizerExample();
  });
}

void runExternallyManagedExample() {
  print('--- 1. Externally Managed Memory (No Custom Finalizer) ---');

  // 1. User allocates raw C memory manually
  const elementCount = 4;
  final ffi.Pointer<ffi.Double> pointer = malloc<ffi.Double>(elementCount);

  try {
    // Initialize memory
    for (var i = 0; i < elementCount; i++) {
      pointer[i] = (i + 1) * 10.0;
    }

    NDArray.scope(() {
      // 2. Wrap raw C memory in NDArray.fromPointer
      // Since we don't pass a nativeFinalizer, this is externally managed.
      final arr = NDArray<DTypeTag>.fromPointer(pointer.cast(), [
        2,
        2,
      ], DType.float64);

      print('Wrapped array data: ${arr.toList()}'); // [10.0, 20.0, 30.0, 40.0]

      // 3. Perform operations/slicing
      final sliced = arr.slice([
        Slice.all(),
        Slice(start: 1),
      ]); // Slice second column
      print('Sliced view data: $sliced');
    });
    print('Array and views automatically disposed at scope exit.');
  } finally {
    // 4. User manually frees the raw memory they allocated
    malloc.free(pointer);
    print('Raw memory manually freed by user.\n');
  }
}

void runCustomFinalizerExample() {
  print('--- 2. Custom Native Finalizer (Self-Managed External Memory) ---');

  NDArray.scope(() {
    // We can use malloc.nativeFree as our custom deallocator function pointer!
    const elementCount = 4;
    final ffi.Pointer<ffi.Double> pointer = malloc<ffi.Double>(elementCount);

    for (var i = 0; i < elementCount; i++) {
      pointer[i] = (i + 1) * 1.5;
    }

    // Wrap with custom nativeFinalizer.
    // The array now owns the deallocation lifecycle via the custom finalizer.
    final arr = NDArray<DTypeTag>.fromPointer(
      pointer.cast(),
      [4],
      DType.float64,
      nativeFinalizer: malloc.nativeFree.cast(),
    );

    print('Custom-finalized array: ${arr.toList()}');
  });
  print('Array and its custom pointer successfully disposed together! 🏆\n');
}
