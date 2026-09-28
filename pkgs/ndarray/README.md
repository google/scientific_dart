# ndarray

<img src="doc/images/mascot.png" align="right" width="220" alt="ndarray mascot">

[![Pub Version](https://img.shields.io/pub/v/ndarray)](https://pub.dev/packages/ndarray)
[![Dart SDK](https://img.shields.io/badge/Dart-%5E3.10.0-blue.svg)](https://dart.dev)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**`package:ndarray`** is a high-performance, strongly-typed N-dimensional array and scientific computing library for Dart, inspired by NumPy and SciPy. Built on Dart Native Assets (`dart:ffi`), it combines an ergonomic, idiomatic Dart API with unmanaged C/C++ memory buffers, Google Highway SIMD vectorization, OpenBLAS/LAPACK linear algebra, and PocketFFT spectral transforms.

---

## Highlights & Architecture

- **Native C/C++ & SIMD Acceleration**: Tensors are stored in raw, contiguous or arbitrarily strided buffers on the unmanaged C heap. Universal element-wise functions (ufuncs), mathematical reductions, and indexing operations compile with an x86-64-v3 baseline (`AVX2`, `FMA`, `F16C`) on x86_64 (configurable via `hooks.user_defines.ndarray.x86Flags` in `pubspec.yaml` or the `NDARRAY_X86_FLAGS` environment variable) and `NEON` on ARM64, completely bypassing Dart VM loop and boxing overhead. On x86_64 hosts, a baseline-compiled CPU check verifies `AVX2`, `FMA`, `F16C`, and `OSXSAVE` support before loading AVX2-compiled kernels and throws an actionable `UnsupportedError` if any required feature is missing.
- **OpenBLAS & LAPACK Linear Algebra**: Hardware-accelerated matrix multiplication (`matmul`), Cholesky factorization (`cholesky`), QR decomposition (`qr`), Singular Value Decomposition (`svd`), symmetric/Hermitian and general eigenvalue problems (`eigh`, `eig`), least-squares solvers (`lstsq`), Moore-Penrose pseudo-inverse (`pinv`), linear system solvers (`solve`), matrix norms, condition numbers (`cond`), and multi-matrix chain multiplication (`multi_dot`).
- **Einstein Summation (`einsum`) & Tensor Contractions**: Express arbitrary multi-dimensional tensor contractions, trace reductions, and batch matrix multiplications concisely using `einsum` (supporting explicit indices and batch ellipsis `...` notation via `EinsumSubscripts.parse('bhid,bhjd->bhij')` or `'...id,...jd->...ij'`) and generalized tensor dot products (`tensordot`), automatically dispatching to optimized BLAS GEMM paths when possible.
- **Google Highway SIMD Sorting & Dynamic Dispatch**: Vectorized sorting and order-statistics routines (`vqsort`) leveraging Google Highway with dynamic multi-target runtime SIMD dispatch (SSE4, AVX2, AVX-512, NEON, SVE) for in-place and out-of-place `sort`, indirect index sorting (`argsort`), and $O(N)$ selection (`partition`, `argpartition`).
- **Spectral Transforms & DSP**: Mixed-radix 1D, 2D, and N-dimensional complex and real Fast Fourier Transforms (`fft`, `ifft`, `rfft`, `irfft`, `fft2`, `fftn`) powered by `package:pocketfft`, complete with frequency bin generators (`fftfreq`, `rfftfreq`), zero-frequency shifting (`fftshift`, `ifftshift`), 1D/2D convolution and cross-correlation (`convolve`, `correlate`), and spectral window functions (Hann, Hamming, Blackman, Bartlett, Kaiser).
- **Numerical Optimization & Root Finding**: Multivariate nonlinear function minimization (`minimize`) supporting quasi-Newton **L-BFGS** (`MinimizeMethod.lbfgs`) and derivative-free **Nelder-Mead** simplex (`MinimizeMethod.nelderMead`), alongside 1D scalar root finding (`root_scalar`) via **Brent's method** (`RootMethod.brentq`), Newton-Raphson, and secant iterations.
- **Spatial Distance Metrics & Orthogonal Polynomials**: Pairwise distance matrix computations (`pdist`, `cdist`, `squareform`) across Euclidean, Manhattan, Cosine, Chebyshev, and Minkowski metrics, plus classical orthogonal polynomial evaluation, fitting, differentiation, integration, and root/quadrature generation (Chebyshev, Legendre, Hermite, Laguerre).
- **Zero-Copy NumPy `.npy` & `.npz` Streaming I/O**: Native binary serialization and deserialization for single arrays (`save`, `load`) and multi-array ZIP archives (`savez`, `loadz` with optional compression), enabling zero-overhead data exchange with Python, NumPy, SciPy, and PyTorch pipelines.
- **Zero-Copy Strided Views & 64-Bit Indexing**: Every `NDArray<T>` pairs an off-heap C buffer with 64-bit N-dimensional `shape` and element `strides` (`int64_t`), supporting large tensors and virtual broadcasted views beyond $2^{31} - 1$ elements with `NDArray<Int64>` index and count operations (`argsort`, `argpartition`, `searchsorted`, `argmax`, `argmin`, `nonzero`, `argwhere`, `count_nonzero`, `digitize`). Operations such as multi-axis slicing (`Slice`), transposing (`.transposed`, `swapaxes`, `moveaxis`), reshaping (`reshape`), dimension expansion (`expand_dims`), and squeezing (`squeeze`) return **zero-copy views** over shared C memory in $O(1)$ time.
- **Deterministic Scoped Memory Management**: Zone-based lexical resource arenas (`NDArray.scope`) and explicit escape hatches (`detachToParentScope`) that automatically track and free unmanaged C-heap allocations deterministically when a computation block completes, eliminating GC pressure and out-of-memory stalls.

---

## Installation & Build Configuration

Add `ndarray` using `dart pub add`:

```bash
dart pub add ndarray
```

Or add it to your `pubspec.yaml`:

```yaml
dependencies:
  ndarray: ^0.2.0
```

### Native Assets, Prebuilt Binaries, & `hooks.user_defines`

`package:ndarray` uses the **Dart Native Assets** build system (`hook/build.dart`) to provide its C/C++ SIMD kernels, PocketFFT, and OpenBLAS/LAPACK bindings automatically when you run `dart run`, `dart test`, or `dart build`.

By default (`buildMode: fetch`), the build hook downloads prebuilt, SLSA-attested shared libraries for your target OS and architecture (`linux-x64`, `linux-arm64`, `macos-arm64`, `macos-x64`, `windows-x64`), verifies their SHA-256 checksums and embedded native source hashes, and falls back to building from source (`buildMode: source`) if prebuilt binaries are unavailable or out of date.

Because the Dart `hooks_runner` isolates build hook subprocesses from arbitrary host environment variables, configure custom build modes, SIMD flags, or C/C++ sanitizers via `hooks.user_defines` in your root `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    ndarray:
      # 'fetch' (default), 'source', or 'local'
      buildMode: source
      # Optional custom x86 SIMD flags (defaults to '-mavx2 -mfma -mf16c' / '/arch:AVX2'):
      x86Flags: "-march=native"
      # Optional Clang/GCC sanitizers ('address', 'undefined', 'address,undefined'):
      sanitize: address,undefined
      # Optional gcov/llvm-cov coverage instrumentation (true / false):
      coverage: false
      # Path to prebuilt binary or directory when buildMode is 'local':
      # localPath: /path/to/dist
    openblas:
      buildMode: source
    pocketfft:
      buildMode: source
```

### Host Compiler Requirements (When Building from Source)

When `buildMode: source` is active (or when `fetch` falls back to source), ensure the following host build tools are installed:

- **Linux**: `cmake`, `make`, and a C/C++ compiler (`gcc`/`g++` or `clang`/`clang++`).
  - *Ubuntu / Debian*: `sudo apt-get update && sudo apt-get install -y build-essential cmake gfortran`
- **macOS**: Xcode Command Line Tools (`xcode-select --install`) and `cmake` (`brew install cmake gcc`).
- **Windows**: Visual Studio 2022 (with the *"Desktop development with C++"* workload / MSVC `cl.exe`) and `cmake` (`winget install Kitware.CMake`).

> [!NOTE]
> **First-Run Source Compilation Time**: When compiling from source for the first time, the build hooks compile OpenBLAS/LAPACK and Google Highway SIMD kernels for your target CPU architecture (approximately **1–2 minutes** depending on CPU core count). Subsequent runs reuse cached artifacts in `.dart_tool/` keyed by compiler flags and source hashes.

---

## Quickstart: Idiomatic Usage & Memory Management

Because `NDArray` allocates raw memory on the native C heap, memory management is deterministic. Wrapping computations in `NDArray.scope(() { ... })` ensures all temporary arrays and intermediate results created inside the block are automatically freed when the scope exits.

To return an array out of a scope without it being disposed, call `.detachToParentScope()` on the result.

```dart
import 'package:ndarray/ndarray.dart';

void main() {
  // Execute a tensor pipeline inside a deterministic memory scope
  final result = NDArray.scope(() {
    // 1. Create a 2x3 Float64 matrix
    final a = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );

    // 2. Create a zero-copy transposed view (shape: [3, 2], O(1) operation)
    final aT = a.transposed;

    // 3. Broadcast addition with a row vector
    final bias = NDArray.fromList([0.5, -0.5], [1, 2], DType.float64);
    final shifted = add(aT, bias);

    // 4. Accelerated OpenBLAS matrix multiplication: [2, 3] x [3, 2] -> [2, 2]
    final product = matmul(a, shifted);

    // Detach 'product' so it survives when this scope exits;
    // 'a', 'aT', 'bias', and 'shifted' are freed automatically!
    return product.detachToParentScope();
  });

  print('Result shape: ${result.shape}'); // [2, 2]
  print('Result data:  ${result.toList()}');

  // Explicitly dispose top-level / detached arrays when no longer needed
  result.dispose();
}
```

> [!WARNING]
> **Python / NumPy User Callout: Equality Comparisons (`equal` / `allClose` vs `==`)**
>
> In Dart, the `==` operator on `NDArray` checks **object identity / exact handle metadata**, and cannot return a boolean array due to Dart language type rules (`bool operator ==(Object other)`).
> - To compute an element-wise boolean mask comparing two arrays (equivalent to NumPy's `a == b`), use **`equal(a, b)`**, which returns an `NDArray<Boolean>`.
> - To check if two floating-point arrays are numerically equal within relative/absolute tolerances ($|a - b| \le \text{atol} + \text{rtol} \cdot |b|$, equivalent to `np.allclose(a, b)`), always use **`allClose(a, b, rtol: 1e-5, atol: 1e-8)`** or **`isClose(a, b)`**.

> [!NOTE]
> **Top-Level Math Functions and `dart:math` Namespace Conflicts**
>
> Because `package:ndarray/ndarray.dart` exports vectorized top-level math functions (`sin`, `cos`, `sqrt`, `max`, `min`, `exp`, `log`, etc.), files that also use `dart:math` for scalar operations should import `dart:math` with a prefix (`import 'dart:math' as math;`) or import `ndarray` with a prefix (`import 'package:ndarray/ndarray.dart' as np;`).

---

## Comprehensive Code Examples

### Example 1: Signal Processing — Vectorized Real FFT Denoising

This example generates a 10 Hz sine wave corrupted by Gaussian sensor noise using vectorized array operations (`NDArray.arange`, `sin`), transforms it to the frequency domain via `rfft`, applies a vectorized low-pass filter using boolean mask assignment (`setByMaskScalar`), and reconstructs the clean time-domain signal using `irfft` (eliminating all scalar `List.generate` loops and boxed list conversions):

```dart
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    const samplingRate = 128.0;
    const numPoints = 128;

    // 1. Vectorized time grid [0.0, 1/128, 2/128, ..., 127/128]
    final time = NDArray.arange(
      0.0,
      numPoints / samplingRate,
      step: 1.0 / samplingRate,
      dtype: DType.float64,
    );

    // 2. Pure 10 Hz sine wave: y(t) = 5.0 * sin(2 * pi * 10.0 * t)
    final angle = time * (2.0 * math.pi * 10.0);
    final pureSignal = sin(angle) * 5.0;

    // 3. Inject Gaussian white noise (sigma = 0.5)
    final noise = normal(
      [numPoints],
      loc: 0.0,
      scale: 0.5,
      dtype: DType.float64,
      seed: 42,
    );
    final noisySignal = add(pureSignal, noise);

    // 4. Compute Real FFT and corresponding frequency bin centers (in Hz)
    final spectrum = rfft(noisySignal);
    final freqs = rfftfreq(numPoints, d: 1.0 / samplingRate);

    // 5. Vectorized low-pass filter: zero out all frequencies above 15 Hz in-place
    final highFreqMask = freqs > 15.0;
    spectrum.setByMaskScalar(highFreqMask, const Complex(0.0, 0.0));

    // 6. Inverse Real FFT back to time domain
    final reconstructed = irfft(spectrum, n: numPoints);

    // 7. Verify accuracy against the original noiseless signal
    final isRecovered = allClose(
      pureSignal,
      reconstructed,
      rtol: 0.3,
      atol: 0.5,
    );
    print('Signal successfully denoised: $isRecovered');
  });
}
```

### Example 2: Machine Learning — Batched Attention & PCA via `einsum` and `svd`

This example demonstrates multi-head scaled dot-product attention scores computed via Einstein summation (`einsum` with `EinsumSubscripts.parse('bhid,bhjd->bhij')`) and Principal Component Analysis (PCA) dimensionality reduction powered by LAPACK Singular Value Decomposition (`svd`):

```dart
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    // --- Part A: Batched Multi-Head Attention Scores via einsum ---
    // Batch size B=2, Heads H=4, Sequence length S=8, Head dim D=16
    final q = normal([2, 4, 8, 16], dtype: DType.float64, seed: 1);
    final k = normal([2, 4, 8, 16], dtype: DType.float64, seed: 2);

    // Compute raw attention logits: Q * K^T across batch and head dimensions
    // (Also supports batch ellipsis notation: '...id,...jd->...ij')
    final rawScores = einsum<Float64, Float64>(
      EinsumSubscripts.parse('bhid,bhjd->bhij'),
      [q, k],
    );
    final scaledScores = rawScores * (1.0 / math.sqrt(16.0));
    print('Batched Attention Scores shape: ${scaledScores.shape}'); // [2, 4, 8, 8]

    // --- Part B: Principal Component Analysis (PCA) via SVD ---
    // Generate synthetic dataset X of shape [50 samples, 6 features]
    final x = normal([50, 6], dtype: DType.float64, seed: 42);

    // Center features by subtracting column-wise mean (keepdims: true -> [1, 6])
    final colMean = mean(x, axis: 0, keepdims: true);
    final xCentered = subtract(x, colMean);

    // Factorize X_centered = U * S * V^T using LAPACK SVD
    final (:u, :s, :vh) = svd(xCentered);

    // Project 6D dataset onto top k=2 principal components (first 2 rows of V^T transposed -> [6, 2])
    const kComponents = 2;
    final principalAxes = vh.slice([
      Slice(stop: kComponents),
      Slice.all(),
    ]).transposed;
    final xProjected = matmul(xCentered, principalAxes);

    print('Original dataset shape:  ${x.shape}'); // [50, 6]
    print('PCA 2D projected shape:  ${xProjected.shape}'); // [50, 2]
    print('Top 2 singular values:   ${s.toList().sublist(0, 2)}');
  });
}
```

### Example 3: Scientific Optimization & NumPy `.npz` Checkpointing

This example minimizes the non-convex Rosenbrock function using quasi-Newton **L-BFGS** (`minimize(..., method: MinimizeMethod.lbfgs)`) and checkpoints the optimization parameters into a compressed NumPy `.npz` archive (`savez` / `loadz`), cleaning up the temporary file in a `try ... finally` block:

```dart
import 'dart:io';
import 'package:ndarray/ndarray.dart';

void main() {
  const checkpointPath = 'rosenbrock_checkpoint.npz';
  final checkpointFile = File(checkpointPath);

  try {
    NDArray.scope(() {
      // Rosenbrock objective function: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
      // Global minimum is at (x, y) = (1.0, 1.0) where f(x, y) = 0.0
      double rosenbrock(NDArray<Float64> params) {
        final x = params.getCell([0]);
        final y = params.getCell([1]);
        final term1 = 1.0 - x;
        final term2 = y - x * x;
        return term1 * term1 + 100.0 * term2 * term2;
      }

      // Initial guess: [-1.2, 1.0]
      final x0 = NDArray.fromList([-1.2, 1.0], [2], DType.float64);

      // Run quasi-Newton L-BFGS optimization
      final result = minimize(
        rosenbrock,
        x0,
        method: MinimizeMethod.lbfgs,
        tol: 1e-8,
      );

      print('Optimization converged: ${result.success}');
      print('Iterations: ${result.nit}, Final loss: ${result.fun.toStringAsExponential(3)}');
      print('Optimal parameters [x, y]: ${result.x.toList()}');

      // Save optimal parameters and initial guess to a NumPy .npz archive
      savez(
        checkpointPath,
        {
          'initial_guess': x0,
          'optimal_params': result.x,
        },
        compressed: true,
      );

      // Reload the checkpoint archive back into memory
      final restored = loadz(checkpointPath);
      final loadedOpt = restored['optimal_params']!;
      print('Reloaded from .npz matches: ${allClose(result.x, loadedOpt)}');
    });
  } finally {
    // Clean up temporary checkpoint file
    if (checkpointFile.existsSync()) {
      checkpointFile.deleteSync();
    }
  }
}
```

---

## Supported Data Types (`DType<T>`)

`ndarray` provides strict compile-time and runtime type safety across **15 numerical and logical data types**:

| `DType` Constant | Type Parameter `T` | Dart Element Type | Byte Width | NumPy Descriptor | Category | Description |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `DType.float64` | `Float64` | `double` | 8 | `<f8` | Floating-Point | Double-precision IEEE 754 float |
| `DType.float32` | `Float32` | `double` | 4 | `<f4` | Floating-Point | Single-precision IEEE 754 float |
| `DType.float16` | `Float16` | `double` | 2 | `<f2` | Floating-Point | Half-precision IEEE 754 float |
| `DType.bfloat16` | `BFloat16` | `double` | 2 | `\|V2` | Floating-Point | Brain Floating-Point (16-bit ML format) |
| `DType.int64` | `Int64` | `int` | 8 | `<i8` | Signed Integer | 64-bit signed two's complement integer |
| `DType.int32` | `Int32` | `int` | 4 | `<i4` | Signed Integer | 32-bit signed two's complement integer |
| `DType.int16` | `Int16` | `int` | 2 | `<i2` | Signed Integer | 16-bit signed two's complement integer |
| `DType.int8` | `Int8` | `int` | 1 | `<i1` | Signed Integer | 8-bit signed two's complement integer |
| `DType.uint64` | `Uint64` | `int` | 8 | `<u8` | Unsigned Integer | 64-bit unsigned integer (use `uint64Compare`) |
| `DType.uint32` | `Uint32` | `int` | 4 | `<u4` | Unsigned Integer | 32-bit unsigned integer |
| `DType.uint16` | `Uint16` | `int` | 2 | `<u2` | Unsigned Integer | 16-bit unsigned integer |
| `DType.uint8` | `Uint8` | `int` | 1 | `\|u1` | Unsigned Integer | 8-bit unsigned byte |
| `DType.complex128` | `Complex128` | `Complex` | 16 | `<c16` | Complex | Double-precision complex (`2 x Float64`) |
| `DType.complex64` | `Complex64` | `Complex` | 8 | `<c8` | Complex | Single-precision complex (`2 x Float32`) |
| `DType.boolean` | `Boolean` | `bool` | 1 | `\|b1` | Boolean / Mask | 8-bit boolean truth value (`0` or `1`) |

---

## Ecosystem & Companion Packages

`ndarray` is the foundation of the Dart Scientific Computing Workspace and integrates with companion packages in this monorepo:

- **[`package:resource_scope`](../resource_scope)**: Zone-based lexical lifetime management (`ResourceScope`) underlying `NDArray.scope` and deterministic FFI memory disposal.
- **[`package:gpuarray`](https://github.com/google/scientific_dart/tree/main/pkgs/gpuarray)** *(experimental)*: GPU-accelerated N-dimensional array computing with compute shaders and zero-copy/streaming `NDArray` interoperability.
- **[`package:ndarray_ma`](../ndarray_ma)** *(experimental)*: Masked arrays (`MaskedArray`) for `ndarray`, enabling robust computation and statistical reductions over datasets with missing or invalid entries.
- **[`package:symbolic_dart`](../symbolic_dart)** *(experimental)*: Native Computer Algebra System (CAS) symbolic mathematics with symbolic-to-numerical `ndarray` evaluation.

### Further Documentation

For deep dives into architecture and usage patterns, see the guides in [`doc/`](doc/):
- **[Memory Management & Scopes Guide](doc/memory_management.md)**: Detailed rules on C-heap lifetimes, views, `NDArray.scope()`, and `detachToParentScope()`.
- **[NumPy to NDArray Quickstart Guide](doc/numpy_quickstart.md)**: Side-by-side translation table and idiomatic patterns for Python/NumPy users.
- **[Shape & Stride Design Choices](doc/design_choice_shape_strides.md)**: How N-dimensional slicing, strides, and memory order are implemented.
- **[Threading & Parallelism Guide](doc/threading.md)**: Multi-threaded OpenBLAS configuration and Dart Isolate concurrency (`SendableNDArray`).

---

## License

This project is licensed under the Apache License, Version 2.0 — see the [LICENSE](https://github.com/google/scientific_dart/blob/main/pkgs/ndarray/LICENSE) file for details.
