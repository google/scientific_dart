# Memory Management with NDArray Scopes

NDArrays are backed by C-heap memory for interoperability with native libraries like OpenBLAS and PocketFFT.

Each root `NDArray` attaches a `dart:ffi` [`NativeFinalizer`](https://api.dart.dev/dart-ffi/NativeFinalizer-class.html) (`calloc.nativeFree`) so its backing C memory is eventually reclaimed when the Dart object becomes unreachable and is garbage collected. However, the Dart garbage collector cannot "feel" the memory pressure of large off-heap allocations because the `NDArray` wrapper on the Dart heap is tiny.

> [!NOTE]
> **Why `externalSize` is intentionally omitted:** `NativeFinalizer.attach` accepts an optional `externalSize` parameter, but it is too blunt a tool—reporting large off-heap buffer sizes can trigger severe GC thrashing on the Dart heap rather than prompt native memory reclamation. Consequently, `NDArray` intentionally omits `externalSize` and relies on `NDArray.scope` (or explicit `.dispose()`) for deterministic lifecycle management, using `NativeFinalizer` strictly as a fallback safety net.

This means memory should be explicitly freed for any serious programs.

To make this safe and ergonomic, `ndarray` provides an **Automatic Disposal Scope** mechanism.

---
## Manual Disposal

Without allocation scopes, you must manually track and dispose of every array you create:

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

## Automatic Disposal with `NDArray.scope`

`NDArray.scope` runs your callback inside a Dart [`Zone`](https://api.dart.dev/dart-async/Zone-class.html) (`dart:async` `runZoned` via `package:resource_scope`) where every root array allocated is automatically registered and deterministically freed when the scope finishes. Because tracking is stored in `Zone.current`, it automatically follows asynchronous calls (`Future`s and `async`/`await` continuations) spawned inside the scope and waits for an `async` callback's returned `Future` to complete before freeing the arrays.

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

If you need a specific array to survive beyond the scope (e.g., as a return value), use `detachToParentScope()`. This unregisters the array from the current scope and attaches it to the enclosing scope (if any).

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

Instead it is often better to pre-allocate a fixed destination array once and pass it to the `out:` named parameter in subsequent operations. This avoids repeated allocations by writing the result directly into the pre-allocated memory:

```dart
void processInLoop(NDArray<Float64> input) {
  // 1. Pre-allocate fixed output buffer once!
  final outputBuffer = NDArray<Float64>.zeros(input.shape, DType.float64);

  for (var i = 0; i < 10000; i++) {
    // 2. Zero new allocations! The ufunc writes directly into the buffer.
    multiply(input, NDArray.scalar(2.0), out: outputBuffer);
    
    // Use the outputBuffer results...
  }
  
  // 3. Free the fixed buffer when completely done
  outputBuffer.dispose();
}
```

To create a buffer of the right shape for the result of a broadcasted operation, you can use [NDArray.broadcastShapes].

---

## Returning Views from Scopes

When returning a **view** (such as a `reshape`, `transpose`, `slice`, or index-view) of an internally allocated array from a scope, the backing memory belongs to the root parent array.

Calling `detachToParentScope()` or `detachFromScope()` directly on a view throws a `StateError` (`Cannot call detachToParentScope() on a view`). This restriction is intentional: allowing a view to be detached directly would secretly keep the entire backing allocation alive in memory to serve a potentially tiny slice, leading to severe native memory leaks.

To return data from a view out of a scope, use one of the two valid patterns:

### 1. Recommended: Compact Copy
Materialize an owning copy of the view and detach the copy. This frees the large temporary parent buffer at scope exit and promotes only the compact slice to the enclosing scope:

```dart
NDArray<Float64> getReshapedSamples() {
  return NDArray.scope(() {
    // 1. Allocate large temporary parent array (registered inside the scope)
    final parent = NDArray<Float64>.create([1000], DType.float64);
    
    // 2. Reshape to get a view
    final view = parent.reshape([10, 100]);
    
    // 3. Materialize a compact owning copy and detach it!
    // The large temporary parent buffer is freed at scope exit.
    return view.copy().detachToParentScope();
  }); 
}
```

### 2. Zero-Copy Root Promotion
If keeping the full parent buffer alive is intentional (e.g. to avoid the cost of copying large buffers), detach the **root owning array** and return the view:

```dart
NDArray<Float64> getReshapedSamplesZeroCopy() {
  return NDArray.scope(() {
    // 1. Allocate parent array
    final largeTemp = NDArray<Float64>.create([1000], DType.float64);
    
    // 2. Reshape to get a view
    final view = largeTemp.reshape([10, 100]);
    
    // 3. Detach the root parent array; return the view!
    largeTemp.detachToParentScope();
    return view;
  });
}
```

---

## Maximum Element Count Limit ($2^{31} - 1$)

Individual `NDArray` allocations support up to **$2^{31} - 1$ (2,147,483,647) total elements**. Attempting to allocate an array whose total element count exceeds this limit throws an `UnsupportedError`.

This limit is inherent to native SIMD and strided kernel indexing: C/C++ ufunc loops, Google Highway vectorization routines, and OpenBLAS/LAPACK matrix routines index strides, offsets, and loop dimensions using 32-bit signed integers (`ffi.Int` / `int32_t`).

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
  final arr = NDArray<Float64>.fromPointer(rawBuffer.cast(), [10, 10], DType.float64);

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
  final arr = NDArray<Float64>.fromPointer(
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
  assert(NDArray.checkNoLeaks());
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



