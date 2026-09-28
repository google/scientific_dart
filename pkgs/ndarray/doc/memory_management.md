# Memory Management with NDArray Scopes

NDArrays are backed by  C-heap memory for its interoperability with native libraries like OpenBLAS and PocketFFT.

NDArrays will reclaim this memory when they are garbage collected. However the garbage collector cannot "feel" the pressure of the array allocations because they are made outside the Dart heap.

This means memory should be explicitly freed for any serious programs.

To make this safe and somewhat ergonomic, `ndarray` provides an **Automatic Disposal Scope** mechanism.

---
Without setting up allocation scopes, you must manually track and dispose of every array you create:

```dart
void calculate() {
  final a = NDArray.zeros([1000], DType.float64);
  try {
    final b = NDArray.ones([1000], DType.float64);
    try {
      final c = add(a, b);
      try {
        print(mean(c));
      } finally {
        c.dispose();
      }
    } finally {
      b.dispose();
    }
  } finally {
    a.dispose();
  }
}
```

This is verbose, prone to leaks if you forget a `dispose()` call, and especially difficult when dealing with intermediate results (like the output of `add(a, b)`).

---

## The Solution: `NDArray.scope`

`NDArray.scope` creates a "safe zone" where every array created is automatically tracked and deterministically freed when the scope finishes.

### Basic Usage

Wrap your logic in `NDArray.scope`:

```dart
void calculate() {
  NDArray.scope(() {
    final a = NDArray.zeros([1000], DType.float64);
    final b = NDArray.ones([1000], DType.float64);
    
    // Intermediate results are also tracked!
    final c = a + b; 
    
    print(mean(c));
  }); // All arrays (a, b, and c) are freed here!
}
```

### Returning Values with `detachToParentScope()`

If you need a specific array to survive beyond the scope (e.g., as a return value), use `attachToParentScope()`. This "unregisters" the array from the current scope.

```dart
NDArray computeResult() {
  return NDArray.scope(() {
    final a = NDArray.arange(0, 10);
    final b = a * 2;
    
    // 'a' will be disposed, but 'b' will be returned attached to the scope created by the caller (if any).
    return b.detachToParentScope();
  });
}
```

> **Note**: You are now responsible for calling `.dispose()` on the returned array when you are finished with it.


## Asynchronous Scopes

The scope mechanism fully supports `async/await`. If your callback returns a `Future`, the scope will wait for that future to complete before disposing of the tracked arrays.

```dart
Future<void> processDataAsync() async {
  await NDArray.scope(() async {
    final data = await loadLargeArray();
    final processed = performComplexMath(data);
    await uploadResults(processed);
  }); // Cleanup happens only after all awaits finish.
}
```

### Avoiding repeated allocations with `out:`

While `NDArray.scope` cleans up intermediate allocations automatically, allocating new arrays inside high-frequency hot loops still incurs heap allocation and garbage collection overhead. 

Instead it is often better to pre-allocate a fixed destination array once and pass it to the `out:` named parameter in subsequent operations. This completely avoids repeated allocations by writing the result directly into the pre-allocated memory:

```dart
void processInLoop(NDArray<Float64> input) {
  // 1. Pre-allocate fixed output buffer once!
  final outputBuffer = NDArray<Float64>.zeros(input.shape, DType.float64);

  for (var i = 0; i < 10000; i++) {
    // 2. Zero new allocations! The ufunc writes directly into the buffer.
    multiply(input, Float64(2.0), out: outputBuffer);
    
    // Use the outputBuffer results...
  }
  
  // 3. Free the fixed buffer when completely done
  outputBuffer.dispose();
}
```

To create a buffer of the right shape for the result of a broadcasted operation, you can use [NDArray.broadcastShapes].

---

## Returning Views from Scopes

When returning a **view** (such as a `reshape`, `transpose`, `slice`, or index-view) of an internally allocated array from a scope, the backing array still belongs to the parent array.

However, the detaching methods (`detachFromScope()` and `detachToParentScope()`) will attach/detach the parent array!

Therefore, you can call detaching methods directly on the returned view, and the system will seamlessly promote the parent memory block, keeping the view fully valid and safe outside the scope:

```dart
NDArray<Float64> getReshapedSamples() {
  return NDArray.scope(() {
    // 1. Allocate parent array (registered inside the scope)
    final parent = NDArray<Float64>.create([1000], DType.float64);
    
    // 2. Reshape to get a view
    final view = parent.reshape([10, 100]);
    
    // 3. Call detach directly on the view!
    // The parent array is automatically detached behind the scenes, keeping the view memory 100% safe!
    return view.detachToParentScope();
  }); 
}
```

---

## User-Allocated (External) Memory

By default, `NDArray` allocations are made on the C heap via `malloc` or `calloc`, and their lifecycles are managed by Dart’s `NativeFinalizer` or `NDArray.scope`. However, there are situations (e.g., calling external native FFI libraries, or using custom C-side arena allocators) where it is much faster or necessary to wrap pre-existing native pointers directly without copying them.

For this purpose, `ndarray` provides the `NDArray.fromPointer` factory constructor.

### Wrapping an External Pointer (Zero-Copy)

You can wrap any raw `ffi.Pointer<ffi.Void>` using `NDArray.fromPointer`. This is a constant-time $O(1)$ operation that performs no data copies, wrapping the pointer directly with Dart TypedData list views.

`NDArray.fromPointer` offers two distinct lifetime management modes:

#### 1. Externally Managed (Default)
If you do **not** supply a `nativeFinalizer` argument, the `NDArray` does **not** own the backing memory:
- Calling `dispose()` on the array will invalidate the array (causing subsequent access attempts to throw a `StateError`) and any views derived from it, but **will not free** the underlying C pointer.
- The user is fully responsible for deallocating the pointer using the appropriate native allocator.
- If the array is garbage collected by Dart, the backing C memory **will not** be automatically freed.

```dart
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';

void processExternalBuffer() {
  // 1. External system/allocator allocates raw C memory
  final ffi.Pointer<ffi.Double> rawBuffer = malloc<ffi.Double>(100);

  // 2. Wrap it in an NDArray. Since no finalizer is passed, this is externally managed.
  final arr = NDArray<double>.fromPointer(rawBuffer.cast(), [10, 10], DType.float64);

  // 3. Run calculations on it zero-copy
  final columnSlice = arr.slice([Slice.all(), Slice(start: 5)]);
  
  // 4. Invalidate Dart-side NDArrays when done
  arr.dispose(); // ColumnSlice is also invalidated

  // 5. External allocator cleans up the raw pointer
  malloc.free(rawBuffer);
}
```

#### 2. Custom Finalization (Self-Managed)
If you pass a native function pointer to the `nativeFinalizer` argument, the `NDArray` **takes ownership** of the pointer's deallocation lifecycle:
- It registers the pointer and function with a Dart `NativeFinalizer`. If the `NDArray` is garbage collected, the finalizer automatically executes the native deallocator.
- Calling `dispose()` manually will detach the finalizer and immediately invoke the deallocation function on the backing pointer.

```dart
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';

void useSelfManagedBuffer() {
  final ffi.Pointer<ffi.Double> rawBuffer = malloc<ffi.Double>(100);

  // Wrap and pass malloc.nativeFree as the custom native finalizer.
  // The array now owns the lifecycle of the buffer!
  final arr = NDArray<double>.fromPointer(
    rawBuffer.cast(),
    [100],
    DType.float64,
    nativeFinalizer: malloc.nativeFree.cast(),
  );

  // Manual dispose detaches the finalizer and frees the raw pointer immediately
  arr.dispose(); 
}
```

### Crucial Safety Guidelines

> [!WARNING]
> Exposing raw pointer wrapping bypasses Dart's standard memory safety boundaries. Always adhere to the following guidelines:
> 1. **No Dart Heap/GC Pinning:** You **cannot** wrap standard Dart-heap lists (e.g., `Float64List`) zero-copy using this factory. Dart's moving GC does not guarantee stable pointer addresses.
> 2. **Avoid Use-After-Free:** If you use the *Externally Managed* mode, ensure the backing C pointer remains valid for the entire duration that the `NDArray` (or any of its views, transposes, or slices) is active.
> 3. **Metadata Integrity:** You must guarantee that the provided `shape` and `strides` match the actual allocated size of the raw memory block. Incorrect metadata will lead to out-of-bounds reads/writes and hard segmentation faults.

---

## Allocation Tracking (Memory Leak Detection)

For debugging and testing, `ndarray` provides an opt-in allocation tracking mode that tracks all memory-allocating (root) `NDArray`s. This allows you to verify that all created arrays have been properly disposed of by the end of your program or test run.

To ensure zero overhead in production, this feature is controlled by a compile-time constant.

### Enabling Tracking

To enable allocation tracking, compile or run your program with the `TRACK_NDARRAY_ALLOCATIONS` environment declaration set to `true`:

```bash
dart --define=TRACK_NDARRAY_ALLOCATIONS=true run bin/main.dart
# Or for tests:
dart --define=TRACK_NDARRAY_ALLOCATIONS=true test --use-data-isolate-strategy
```

### Checking for Leaks

When tracking is enabled, you can use the static methods on `NDArray` to inspect active allocations and assert that everything has been disposed:

```dart
import 'package:ndarray/ndarray.dart';

void main() {
  // If compiled with TRACK_NDARRAY_ALLOCATIONS=true
  if (NDArray.trackAllocations) {
    print('Tracking is active!');
  }

  final a = NDArray.zeros([100], DType.float64);

  // Retrieve currently active root allocations
  print(NDArray.trackedAllocations.length); // Prints 1

  a.dispose();

  print(NDArray.trackedAllocations.length); // Prints 0

  // Throws StateError if there are any undisposed tracked arrays
  NDArray.checkNoLeaks(); 
}
```

If `NDArray.checkNoLeaks()` is called and there are leaks, it throws a `StateError` with details about the leaked arrays:

```
StateError: Detected 1 undisposed NDArrays:
  NDArray(shape: [100], dtype: Float64)
```

You can also clear the tracked allocations manually using `NDArray.clearTrackedAllocations()`.

### Zero Overhead in Production

When `TRACK_NDARRAY_ALLOCATIONS` is not defined (or set to `false`), the tracking code is completely optimized away by the Dart compiler, resulting in exactly zero runtime overhead.



