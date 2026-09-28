# Threading and Concurrency in NDArray

This guide explains how multi-threading and isolate concurrency work in `ndarray`, how to configure underlying BLAS threads, and how to safely pass data across Dart Isolates using `SendableNDArray`.

---

## 1. OpenBLAS Multi-Threading

### Default Behavior
By default, the `ndarray` package automatically configures the underlying OpenBLAS library to run in **single-threaded mode** (`1` thread) process-wide.
* This configuration is applied automatically upon the first allocation of any `NDArray`.
* Single-threaded mode is highly optimized for small-to-medium matrix operations by avoiding the high CPU overhead associated with spawning and synchronizing thread pools.

### Overriding the Default
You can override the process-wide thread count at any time using the public `setNumThreads` function:

```dart
import 'package:ndarray/ndarray.dart';

void main() {
  // Configure OpenBLAS to use 4 threads for heavy computations
  setNumThreads(4);

  // ... subsequent operations use the configured thread count ...
}
```

### Multi-Threading (> 1 thread) vs. Single-Threading (1 thread)
* **Using Multi-Threading (> 1 thread)**:
  * **Benefits**: Significant speedup for massive linear algebra operations (e.g., multiplying matrices larger than $1000 \times 1000$) by exploiting parallel CPU cores.
  * **Risks (Deadlocks)**: OpenBLAS is not safe to initialize or run in parallel thread pools across multiple Dart Isolates. **If you increase the thread count above `1`, you must not invoke FFI math functions concurrently from multiple Dart Isolates.** Doing so can trigger C-level mutex deadlocks, causing the entire Dart process to hang indefinitely.
* **Using Single-Threading (Default, 1 thread)**:
  * **Benefits**: Complete isolate thread-safety. You can safely run computations concurrently across any number of Dart Isolates (e.g. parallel test suites, concurrent background isolate tasks) without hangs.
  * **Drawbacks**: Massive computations (such as very large matrix multiplications) are restricted to a single CPU core and will run slower compared to a native multi-threaded BLAS execution.

---

## 2. Dart Isolate Concurrency & `SendableNDArray`

### Why `NDArray<T>` Cannot Be Sent Directly
In Dart, an `NDArray<T>` cannot be sent directly across isolates (via `Isolate.run` or `SendPort.send`) due to fundamental runtime safety boundaries:
1. **`NativeFinalizer` Bindings**: Each owning `NDArray` registers a native finalizer token with the isolate-local GC to prevent memory leaks. Dart prohibits sending objects attached to isolate-bound finalizers across isolate ports.
2. **`ResourceScope` Zones**: `NDArray` instances are tracked within zone-scoped resource registries (`NDArray.scope`). These zones and parent view chains (`_parent`) are local to the originating isolate.
3. **Double-Free & Race Conditions**: If multiple isolates held owning references with native finalizers to the same unmanaged C pointer, garbage collection in one isolate would free the pointer while another isolate is actively reading or writing, crashing the Dart VM.

To solve this, `ndarray` provides `SendableNDArray<T>`, which offers two explicit transmission modes:
- **Copy Mode (`toSendable()` / `SendableNDArray.fromCopy`)**: Transferable deep copy.
- **Borrow Mode (`toSendableBorrow()` / `SendableNDArray.unsafeBorrow`)**: Zero-copy shared native memory view.

---

### Mode A: Transferable Copy (`toSendable` / `materialize`)

Copy mode copies the array elements into a Dart VM `TransferableTypedData` buffer. This decouples the memory from the source isolate's lifetime and allows the receiving isolate to reconstruct an independent, fully owned, scope-registered `NDArray<T>`.

#### When to Use
- Passing inputs to background worker tasks where the worker should own the array.
- Returning results computed on a background isolate back to the main isolate.
- Long-running worker tasks that outlive the caller's scope.

#### Example: Offloading Computations to `Isolate.run`
```dart
import 'dart:isolate';
import 'package:ndarray/ndarray.dart';

Future<void> main() async {
  await NDArray.scope(() async {
    // 1. Create source array on the main isolate
    final a = NDArray<Float64>.ones([1000, 1000], DType.float64);

    // 2. Package into a transferable copy ($O(N)$ copy)
    final sendableA = a.toSendable();

    // 3. Offload computation to a worker isolate
    final resultSendable = await Isolate.run(() {
      return NDArray.scope(() {
        // Enforce single-threaded OpenBLAS in worker isolate
        setNumThreads(1);

        // Materialize reconstructs a fresh, scope-registered NDArray
        final workerA = sendableA.materialize();
        final squared = multiply(workerA, workerA);

        // Package result to send back to parent isolate
        return squared.toSendable();
      });
    });

    // 4. Materialize result on the main isolate
    final result = resultSendable.materialize();
    print('Result shape: ${result.shape}');
  });
}
```

---

### Mode B: Zero-Copy Shared Memory Borrowing (`toSendableBorrow` / `materializeView`)

Borrow mode captures the raw native memory address (`address`), `shape`, `strides`, `dtypeIndex`, and `physicalByteCapacity` without copying memory ($O(1)$ complexity). The worker isolate calls `materializeView()` to create a non-owning `NDArray` view backed by `NDArray.fromPointer(..., nativeFinalizer: null)`.

#### Safety Contract
Because borrowed views bypass isolate memory isolation:
1. **Lifetime Guarantee**: The source array **must remain alive and undisposed** on the sending isolate for the entire duration that the worker isolate executes. Always await the `Future` returned by `Isolate.run` within the scope that owns the source array.
2. **Synchronization**: Do not concurrently write to overlapping memory regions from multiple isolates without explicit synchronization.
3. **No Ownership**: The materialized view on the worker isolate does not own the memory and will not free the underlying C pointer on `dispose()`.

#### Example: In-Place Mutation Across Isolates
```dart
import 'dart:isolate';
import 'package:ndarray/ndarray.dart';

Future<void> main() async {
  await NDArray.scope(() async {
    // 1. Allocate buffer on the main isolate
    final output = NDArray<Float64>.zeros([1000], DType.float64);

    // 2. Borrow the raw native pointer (O(1), zero-copy)
    final borrowedOutput = output.toSendableBorrow();

    // 3. Worker mutates memory in-place
    await Isolate.run(() {
      NDArray.scope(() {
        setNumThreads(1);

        // Construct zero-copy non-owning view over shared C memory
        final view = borrowedOutput.materializeView();

        // Perform in-place mutation
        for (var i = 0; i < view.shape[0]; i++) {
          view[i] = (i * 2.5) as Float64;
        }
      });
    });

    // 4. Main isolate immediately observes the updated memory
    print('First element: ${output[0]}'); // 0.0
    print('Second element: ${output[1]}'); // 2.5
  });
}
```

---

## 3. Best Practices for Isolate Concurrency

1. **Always Set `setNumThreads(1)` in Worker Isolates**:
   When launching background workers via `Isolate.run`, invoke `setNumThreads(1)` at the start of the worker callback. This prevents multiple isolates from concurrently triggering multi-threaded OpenBLAS routines, avoiding C-level thread pool conflicts and deadlocks.
2. **Combine Scopes with Isolate Tasks**:
   Wrap worker operations in `NDArray.scope(() { ... })`. Any intermediate arrays allocated by the worker will be deterministically cleaned up when the worker completes.
3. **Choose the Right Mode**:
   - Use `toSendable()` when data ownership transfers across isolates or when arrays outlive the caller's scope.
   - Use `toSendableBorrow()` for massive arrays where copying is cost-prohibitive and the worker executes synchronously within the lifetime of the parent isolate's `await`.
