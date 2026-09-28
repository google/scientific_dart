# Threading and Concurrency in NDArray

## Default Behavior
By default, the `ndarray` package automatically configures the underlying OpenBLAS library to run in **single-threaded mode** (`1` thread) process-wide.
* This configuration is applied automatically upon the first allocation of any `NDArray`.
* Single-threaded mode is highly optimized for small-to-medium matrix operations by avoiding the high CPU overhead associated with spawning and synchronizing thread pools.

## Overriding the Default
You can override the process-wide thread count at any time using the public `setNumThreads` function:

```dart
import 'package:ndarray/ndarray.dart';

void main() {
  // Configure OpenBLAS to use 4 threads for heavy computations
  setNumThreads(4);

  // ... subsequent operations use the configured thread count ...
}
```

### 1. Using Multi-Threading (> 1 thread)
* **Benefits**: Significant speedup for massive linear algebra operations (e.g., multiplying matrices larger than $1000 \times 1000$) by exploiting parallel CPU cores.
* **Risks (Deadlocks)**: OpenBLAS is not safe to initialize or run in parallel thread pools across multiple Dart Isolates. **If you increase the thread count above `1`, you must not invoke FFI math functions concurrently from multiple Dart Isolates.** Doing so can trigger C-level mutex deadlocks, causing the entire Dart process to hang indefinitely.

### 2. Using Single-Threading (Default, 1 thread)
* **Benefits**: Complete isolate thread-safety. You can safely run computations concurrently across any number of Dart Isolates (e.g. parallel test suites, concurrent background isolate tasks) without hangs.
* **Drawbacks**: Massive computations (such as very large matrix multiplications) are restricted to a single CPU core and will run slower compared to a native multi-threaded BLAS execution.
