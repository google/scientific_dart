# NumPy to `package:ndarray` Quickstart & Translation Guide

**Confidence: High** — Verified directly against `package:ndarray` source architecture, FFI bindings, and zone-scoped memory model.

This guide translates the official **NumPy Quickstart / Fundamentals tutorial** into Dart **`package:ndarray`** idioms. While `package:ndarray` adopts NumPy's API philosophy, multi-dimensional stride calculations, broadcasting rules, and ufunc vectorization, it operates within Dart's strong static type system and manual C-heap FFI memory model.

---

## 1. Executive Rosetta Stone: Paradigm Differences

Before diving into code, understand the five structural differences between Python's NumPy and Dart's `package:ndarray`:

1. **Unmanaged C-Heap Allocations & Zone Scopes (`NDArray.scope`)**:
   - **NumPy**: Arrays are Python objects backed by C pointers; Python's reference counting and GC automatically collect them.
   - **`package:ndarray`**: Array buffers are allocated directly on the unmanaged C heap (`malloc`/`calloc`) to enable zero-overhead FFI calls to OpenBLAS, LAPACK, and PocketFFT kernels. Relying on the Dart GC alone can lead to native memory saturation because Dart is blind to large unmanaged C heap sizes.
   - **Rule**: Always wrap computations in `NDArray.scope(() { ... })` to automatically dispose intermediate arrays, or explicitly call `.dispose()`. Use `NDArray.returning(() { ... })` or `.detachFromScope()` / `.detachToParentScope()` when returning arrays out of a scope.

2. **Static `DType` Typing (`NDArray<Float64>`, `NDArray<Int32>`, etc.)**:
   - **NumPy**: `np.ndarray` is dynamically typed; `float64_arr + int32_arr` promotes automatically at runtime.
   - **`package:ndarray`**: Arrays carry their data type in their Dart type (`NDArray<Float64>`, `NDArray<Float32>`, `NDArray<Int32>`, `NDArray<Boolean>`), inferred from `DType.float64`, `DType.int32`, etc.
     - Standard operators (`a + b`, `add(a, b)`) require both arrays to have the **same** type.
     - To combine different types, use `*As` (`addAs(f64, i32, DType.float64)`) or convert explicitly (`f64 + i32.astype(DType.float64)`). See **[Section 2.C.1: Mixed-Type Arithmetic](#c1-mixed-type-arithmetic-float64--int32-float32--float64-etc)**.

3. **Explicit Statically-Typed Access vs. Polymorphic Overloads**:
   - **NumPy**: Square brackets `arr[...]` handle scalar indexing, slicing, boolean masking, and fancy indexing.
   - **`package:ndarray`**: Overloads operator `[]` and `[]=` for NumPy-like polymorphic indexing, **plus** provides zero-overhead, statically-typed explicit accessors that return and accept the corresponding Dart type (`double`, `int`, `bool`, or `Complex`):
     - Multi-dimensional cell read/write: `.getCell([row, col])` and `.setCell([row, col], val)`.
     - 0-Dimensional scalar access: `.scalar` getter (prefer `.scalar` or `.toList()` over indexing `.data` directly, as `.data` is a raw 1-D buffer view that bypasses shapes, strides, and view offsets).
     - Explicit mutations: `.setByMask(mask, vals)`, `.setByMaskScalar(mask, scalarVal)`, `.setIndices(indices, vals)`.

4. **Strongly-Typed Enums over Magic Strings**:
   - **NumPy**: Uses magic string options (e.g., `side='left'`, `method='weibull'`).
   - **`package:ndarray`**: Enforces Dart enums (`SearchSide.left`, `QuantileMethod.weibull`, `SortKind.quicksort`) for compile-time safety.

5. **Integer Division by Zero Safety (`~/` and `%`)**:
   - **NumPy / C++**: Division by zero on integer arrays is undefined C behavior (`SIGFPE` crash).
   - **`package:ndarray`**: True floating-point division (`/`) follows IEEE 754 (`double.nan`, `double.infinity`) and automatically promotes integer arrays (`NDArray<Int32> / NDArray<Int32>` $\to$ `NDArray<Float64>`). Integer floor division (`~/`) and remainder (`%`) check for `0` divisors in the C kernel and throw an explicit `UnsupportedError('Integer division by zero')` to prevent uncatchable `SIGFPE` process crashes, since integer types lack representable `NaN` or `Infinity` states.

6. **Top-Level Math Functions and `dart:math` Namespace Conflicts**:
   - **NumPy**: Typically imported as `import numpy as np`, namespacing math functions under `np.sin`, `np.cos`, `np.sqrt`, etc.
   - **`package:ndarray`**: Exports top-level vectorized math functions (`sin`, `cos`, `sqrt`, `max`, `min`, `exp`, `log`, etc.) directly from `package:ndarray/ndarray.dart`. Files that also use `dart:math` for scalar operations should import `dart:math` with a prefix (`import 'dart:math' as math;`) or import `ndarray` with a prefix (`import 'package:ndarray/ndarray.dart' as np;`) to avoid identifier collisions.

---

## 2. Comprehensive Quickstart Cheat Sheet: NumPy vs. Dart `package:ndarray`

### A. Array Creation & Generators

| NumPy (`import numpy as np`) | Dart `package:ndarray` (`import 'package:ndarray/ndarray.dart'`) | Notes / Differences |
| :--- | :--- | :--- |
| `np.array([1.0, 2.0, 3.0])` | `NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64)` | Infers `NDArray<Float64>` from `DType.float64`. |
| `np.array([[1, 2], [3, 4]])` | `NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32)` | Infers `NDArray<Int32>` from `DType.int32`. |
| `np.zeros((2, 3), dtype=np.float64)` | `NDArray.zeros([2, 3], DType.float64)` | Allocates zero-initialized C memory via `calloc`. |
| `np.ones((2, 3), dtype=np.float32)` | `NDArray.ones([2, 3], DType.float32)` | Fills array with `1.0`. |
| `np.full((2, 2), 42)` | `NDArray.full([2, 2], 42, dtype: DType.int32)` | Fills array with specified scalar value. |
| `np.arange(0, 10, 2)` | `NDArray.arange(0.0, 10.0, step: 2.0, dtype: DType.float64)` | Generates range with step size. |
| `np.linspace(0, 1, 5)` | `linspace(0.0, 1.0, 5, dtype: DType.float64)` | Returns `numSamples` evenly spaced samples. |
| `np.eye(3)` | `NDArray.eye(3, DType.float64)` | Creates `n x n` identity matrix. |
| `np.array(42)` (0-D scalar) | `NDArray.scalar(42, dtype: DType.int32)` | Creates a 0-D array with empty shape `[]`. Retrieve via `.scalar`. |
| `np.random.normal(0, 1, (3, 3))` | `normal([3, 3], loc: 0.0, scale: 1.0)` | Generates Gaussian RNG sample array (`NDArray<Float64>`). |

---

### B. Array Attributes & Properties

| NumPy Property | Dart `NDArray` Property | Example / Meaning |
| :--- | :--- | :--- |
| `arr.ndim` | `arr.rank` | Number of dimensions (e.g., `2` for a matrix). |
| `arr.shape` | `arr.shape` | List/tuple of dimension sizes (e.g., `[2, 3]`). |
| `arr.size` | `arr.size` | Total element count (product of shape dimensions). |
| `arr.dtype` | `arr.dtype` | Stored element descriptor (e.g., `DType.float64`). |
| `arr.strides` | `arr.strides` | Step size in elements along each dimension. |
| `arr.flags['C_CONTIGUOUS']` | `arr.isContiguous` | `true` if elements are stored sequentially in C memory. |
| `arr.base is not None` | `arr.isView` | `true` if array shares memory with a parent array. |
| *None* | `arr.isSquare` | `true` if array has rank 2 and equal row/column count. |

---

### C. Basic Arithmetic & Universal Functions (ufuncs)

| NumPy Operation | Dart `package:ndarray` Equivalent | Description |
| :--- | :--- | :--- |
| `a + b`, `np.add(a, b)` | `a + b` or `add(a, b)` (same dtype) / `addAs(a, b, DType.float64)` (mixed) | Element-wise addition with broadcasting. |
| `a - b`, `np.subtract(a, b)`| `a - b` or `subtract(a, b)` (same dtype) / `subtractAs(a, b, DType.float64)` | Element-wise subtraction with broadcasting. |
| `a * b`, `np.multiply(a, b)`| `a * b` or `multiply(a, b)` (same dtype) / `multiplyAs(a, b, DType.float64)` | Element-wise Hadamard multiplication. |
| `a / b`, `np.divide(a, b)` | `a / b` or `divide(a, b)` (auto-promotes ints to `Float64`) / `divideAs(...)` | True IEEE 754 floating-point division (`nan`/`inf`). |
| `a // b` | `a ~/ b` or `floorDivide(a, b)` / `floorDivideAs(a, b, ...)` | Floor integer/float division (checks `0` divisor upfront). |
| `a % b`, `np.remainder(a,b)`| `a % b` or `remainder(a, b)` / `remainderAs(a, b, ...)` | Element-wise remainder. |
| `-a` | `-a` or `negative(a)` | Element-wise negation. |
| `a @ b`, `np.matmul(a, b)` | `matmul(a, b)` (same dtype) / `matmulAs(a, b, DType.float64)` (mixed) | OpenBLAS / LAPACK matrix multiplication. |
| `np.sin(a)`, `np.cos(a)` | `sin(a)`, `cos(a)`, `tan(a)` | Trigonometric universal functions (ints infer `NDArray<Float64>`). |
| `np.exp(a)`, `np.log(a)` | `exp(a)`, `log(a)`, `log10(a)` | Exponential & logarithmic ufuncs (ints infer `NDArray<Float64>`). |
| `np.sqrt(a)`, `np.power(a,2)`| `sqrt(a)`, `power(a, b)`, `floatPower(a, b)` | Square root and exponentiation. |
| `np.nan_to_num(a, nan=0)` | `nan_to_num(a, nan: 0.0)` | Replaces NaNs and infinities with specified numbers. |

---

### C.1. Mixed-Type Arithmetic (`Float64` + `Int32`, `Float32` + `Float64`, etc.)

In Python/NumPy, `a + b` inspects the two operands' dtypes at runtime (`np.result_type(a, b)`) and dynamically chooses the output dtype.

In Dart, `NDArray<T>` is statically typed by its element type tag (`Float64`, `Float32`, `Int32`, `Complex128`, etc.). The standard same-type operators (`a + b`, `add(a, b)`, `matmul(a, b)`) require both arrays to have the **same** type parameter `T` so that `add(f64a, f64b)` statically returns `NDArray<Float64>` without requiring explicit type arguments.

When combining arrays of **different** dtypes, choose one of the four patterns below depending on your use case:

#### 1. Single-Pass Mixed-Type Kernels via `*As` (`addAs`, `subtractAs`, `multiplyAs`, `divideAs`, `matmulAs`) — *Fastest (Zero Copy)*
Every binary operation and linear algebra contraction has a `*As` counterpart (`addAs`, `subtractAs`, `multiplyAs`, `divideAs`, `floorDivideAs`, `remainderAs`, `powerAs`, `maximumAs`, `minimumAs`, `matmulAs`, `dotAs`, `tensordotAs`, etc.).
Under the hood, `package:ndarray`'s C++/Highway SIMD kernels natively accept mixed input pointer types and promote on the fly in registers—**without allocating a temporary converted copy of either input array**:

```dart
final f64 = NDArray.fromList([1.5, 2.5, 3.5], [3], DType.float64); // NDArray<Float64>
final i32 = NDArray.fromList([10, 20, 30], [3], DType.int32);      // NDArray<Int32>

// Pattern 1A: Specify target `dtype` — Dart infers `NDArray<Float64>` automatically!
final sum1 = addAs(f64, i32, DType.float64); // static type: NDArray<Float64>

// Pattern 1B: Provide a pre-allocated `out:` buffer — zero allocations
final out = NDArray.empty([3], DType.float64);
addAs(f64, i32, DType.float64, out: out);
```

#### 2. Explicit `.astype(...)` Conversion with Infix Operators (`+`, `-`, `*`)
When writing chained mathematical expressions with infix operators (`+`, `-`, `*`), promote the narrower operand explicitly with `.astype(...)` inside an `NDArray.scope` (which automatically frees the temporary converted array when the scope exits):

```dart
final result = NDArray.scope(() {
  final res = (f64 + i32.astype(DType.float64)) * 2.0; // static type: NDArray<Float64>
  return res.detachToParentScope();
});
```

#### 3. Array + Dart Scalar Arithmetic (`arr + 2.0`, `arr * 5`)
Arithmetic and comparison operators between an `NDArray<T>` and a Dart scalar (`num` or `Complex`) work directly without converting the array, preserving the array's static type `NDArray<T>`:

```dart
final f32 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
final scaled = (f32 * 2) + 0.5; // static type remains NDArray<Float32>
```

#### 4. Automatic Float Promotion for True Division (`/`) and Unary Math (`sin`, `exp`, `mean`)
Single-operand functions and true division (`/`) that promote integer arrays to floating-point in NumPy automatically infer the promoted return type at compile time:

```dart
final i32 = NDArray.fromList([1, 2, 4], [3], DType.int32);   // NDArray<Int32>
final f32 = NDArray.fromList([1.0, 2.0], [2], DType.float32); // NDArray<Float32>

// Integer arrays automatically promote to NDArray<Float64>:
final quotI32 = i32 / i32;     // inferred static type: NDArray<Float64>
final sinI32  = sin(i32);      // inferred static type: NDArray<Float64>
final meanI32 = mean(i32);     // inferred static type: NDArray<Float64>

// Float32 arrays preserve NDArray<Float32> for element-wise ops (mean promotes to Float64):
final quotF32 = f32 / f32;     // inferred static type: NDArray<Float32>
final sinF32  = sin(f32);      // inferred static type: NDArray<Float32>
final meanF32 = mean(f32);     // inferred static type: NDArray<Float64>
```

---

### D. Comparisons & Equality Testing

| NumPy Operation | Dart `package:ndarray` Equivalent | Notes |
| :--- | :--- | :--- |
| `a > b`, `a <= b` | `a > b`, `a <= b` | Returns boolean mask `NDArray<Boolean>`. |
| `a == b`, `a != b` | `equal(a, b)`, `notEqual(a, b)` | **IMPORTANT**: Dart `a == b` tests object identity (`bool`). Use `equal(a, b)` for element-wise `NDArray<Boolean>` comparison! |
| `np.isclose(a, b)` | `isClose(a, b, rtol: 1e-5, atol: 1e-8)` | Element-wise approximate floating-point comparison (`NDArray<Boolean>`). |
| `np.allclose(a, b)` | `allClose(a, b, rtol: 1e-5)` | Returns a single `bool` if all elements match within tolerance. |

---

### E. Indexing, Slicing & Masking

| NumPy Syntax | Dart Polymorphic Syntax (`[]`) | Dart Explicit Statically-Typed Accessor |
| :--- | :--- | :--- |
| `val = arr[0, 1]` | `arr[[0, 1]]` | `arr.getCell([0, 1])` *(Recommended — returns `double`/`int`/`bool`/`Complex`)* |
| `arr[1, 0] = 99` | `arr[[1, 0]] = 99` | `arr.setCell([1, 0], 99)` *(Recommended — accepts `double`/`int`/`bool`/`Complex`)* |
| `row = arr[1]` | `arr[1]` | `slice([Index(1)])` |
| `sub = arr[1:3, :]` | `arr[[Slice(start: 1, stop: 3), Slice.all()]]` | `slice([Slice(start: 1, stop: 3), Slice.all()])` |
| `fancy = arr[[0, 2]]` | `arr[[ [0, 2] ]]` (rows) | `take([0, 2])` |
| `arr[arr < 0] = 0` | `arr[arr < 0] = 0` | `arr.setByMaskScalar(arr < 0, 0.0)` |
| `filtered = arr[mask]` | `arr[mask]` | `applyMask(arr, mask)` |

---

### F. Shape Manipulation, Views & Copies

| NumPy Method | Dart Method | Zero-Copy View or Deep Copy? |
| :--- | :--- | :--- |
| `arr.reshape(2, 5)` | `arr.reshape([2, 5])` | **Zero-Copy View** if `arr.isContiguous`; deep copy if strided. |
| `arr.T`, `arr.transpose()`| `arr.transposed`, `arr.transpose()`| **Zero-Copy View** with permuted strides (`[C, B, A]`). |
| `np.expand_dims(a, 0)` | `expand_dims(a, 0)` | **Zero-Copy View** inserting size-1 axis at position `0`. |
| `np.squeeze(a, axis=0)` | `squeeze(a, axis: [0])` | **Zero-Copy View** dropping size-1 dimensions. |
| `arr.flatten()` | `arr.flatten()` | **Deep 1-D Copy** (`arr.flatten()` always returns an independent 1-D copy). |
| `arr.ravel()` | `arr.ravel()` | **Zero-Copy View** (`arr.ravel()` returns a zero-copy view when contiguous, else copy). |
| `arr.copy()` | `arr.copy()` | **Deep C-Contiguous Copy** (decouples backing C memory). |

---

### G. Stacking, Concatenating & Splitting

- **Concatenation (`concatenate`)**: Joins arrays along an **existing** dimension (rank unchanged).
  ```dart
  // a: shape [2], b: shape [2] -> result: shape [4]
  final c = concatenate([a, b], axis: 0);
  ```
- **Stacking (`stack`)**: Joins arrays along a **brand new** dimension (increases rank by 1).
  ```dart
  // a: shape [3], b: shape [3] -> result: shape [2, 3]
  final s = stack([a, b], axis: 0);
  ```
- **Splitting**:
  - `split(a, 2)`: Splits array `a` into `2` **equal** sections along axis 0. (Throws if size not divisible).
  - `split_at(a, [2, 5])`: Splits array `a` at specific **index split points** `2` and `5`.
  - `array_split(a, 3)`: Splits array `a` into `3` sections, distributing remainder elements as evenly as possible.
  - `hsplit(a, 2)` / `vsplit(a, 2)` / `dsplit(a, 2)`: Column-wise (axis 1), row-wise (axis 0), and depth-wise (axis 2) equal splits.

---

### H. Statistical Reductions (`axis` and `keepdims`)

In `package:ndarray`, reduction functions take an optional `axis` integer and `bool keepdims = false`.
If `axis` is omitted, the array is reduced across all dimensions to a **0-D scalar array**:

```dart
final mat = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

// Total sum (0-D scalar array -> retrieve value with .scalar)
final totalSum = sum(mat);
print(totalSum.scalar); // 10.0

// Sum along columns (axis 0, collapses rows) -> shape [2]
final colSum = sum(mat, axis: 0);
print(colSum.toList()); // [4.0, 6.0]

// Mean along rows preserving rank (axis 1, keepdims: true) -> shape [2, 1]
final rowMean = mean(mat, axis: 1, keepdims: true);
print(rowMean.shape); // [2, 1]
```

- **Available Reductions**: `sum`, `mean`, `prod`, `variance` / `std`, `min`, `max`, `argmin`, `argmax`.
- **NaN-Safe Reductions**: `nansum`, `nanmean`, `nanmin`, `nanmax`, `nanvar`, `nanstd`.

---

### I. Advanced Linear Algebra & FFTs

`package:ndarray` accelerates linear algebra operations via OpenBLAS/LAPACK and Fourier transforms via PocketFFT:

```dart
// Matrix multiplication (OpenBLAS DGEMM / SGEMM)
final A = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.float64);
final B = NDArray.fromList([5, 6, 7, 8], [2, 2], DType.float64);
final C = matmul(A, B);

// Determinant & Matrix Inversion (LAPACK DGETRF / DGETRI)
final detA = det(A);
final invA = inv(A);

// Solve linear system A * x = b (LAPACK DGESV)
final b = NDArray.fromList([10.0, 20.0], [2, 1], DType.float64);
final x = solve(A, b);

// Fast Fourier Transform & Shift (PocketFFT)
final signal = normal([128], loc: 0.0, scale: 1.0);
final spectrum = fft(signal);
final shifted = fftshift(spectrum);
```

---

## 3. Tutorial Walkthrough: 1D & 2D Operations with Automatic Memory Management

Here is a self-contained tutorial snippet demonstrating array creation, slicing, ufuncs, reduction, and scope-based memory cleanup:

```dart
import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Quickstart Walkthrough ===');

  // Rule #1: Use NDArray.scope to manage C-heap lifecycles!
  NDArray.scope(() {
    // 1. Array Creation
    final a = NDArray.arange(0.0, 12.0, step: 1.0, dtype: DType.float64);
    final mat = a.reshape([3, 4]);
    print('Matrix shape: ${mat.shape}, strides: ${mat.strides}');

    // 2. Explicit Statically Typed Cell Access (inferred double from Float64)
    mat.setCell([1, 2], 999.0);
    print('Cell [1, 2] modified to: ${mat.getCell([1, 2])}');

    // 3. Zero-Copy Transpose View
    final transposed = mat.transposed;
    print('Is transposed a view? ${transposed.isView}'); // true
    print('Transposed shape: ${transposed.shape}');

    // 4. Element-wise Math & ufuncs
    final scaled = (mat * 2.0) + 10.0;
    final sinMat = sin(scaled);

    // 5. Mixed-Type Arithmetic (Float64 + Int32 -> Float64 via addAs)
    final offsets = NDArray.fromList([1, 2, 3, 4], [1, 4], DType.int32);
    final shifted = addAs(scaled, offsets, DType.float64);

    // 6. Reductions & Comparison Masking
    final rowSums = sum(shifted, axis: 1);
    print('Row sums (axis 1): ${rowSums.toList()}');

    // Boolean filtering: zero out elements < 50
    final mask = shifted < 50.0;
    shifted.setByMaskScalar(mask, 0.0);
    print('After clipping < 50 to 0:\n$shifted');
  });
}
```
