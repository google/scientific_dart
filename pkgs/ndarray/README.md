# ndarray

<img src="doc/images/mascot.png" align="right" width="220" alt="ndarray mascot">

[![Pub Version](https://img.shields.io/pub/v/ndarray)](https://pub.dev/packages/ndarray)
[![Dart SDK](https://img.shields.io/badge/Dart-%5E3.10.0-blue.svg)](https://dart.dev)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A strongly-typed N-dimensional array and scientific computing library for Dart, inspired by NumPy.

Uses Dart Native Assets (`dart:ffi`) to provide an ergonomic, idiomatic Dart API over C/C++ memory buffers, Google Highway SIMD vectorization, OpenBLAS/LAPACK linear algebra, and PocketFFT spectral transforms.

> **Live Demo:** Try the **[Scientific Dart Notebook in your browser](https://google.github.io/scientific_dart/)** — runs `ndarray`, `openblas`, `pocketfft`, and `gpuarray` entirely client-side via WebAssembly and WebGPU.

---

## Highlights & Architecture

- **Native C/C++ & SIMD Acceleration**: Tensors are stored in raw, contiguous or arbitrarily strided buffers on the unmanaged C heap. Universal element-wise functions (ufuncs), mathematical reductions, and indexing operations compile with an x86-64-v3 baseline (`AVX2`, `FMA`, `F16C`) on x86_64 (configurable via `hooks.user_defines.ndarray.x86Flags` in `pubspec.yaml` or the `NDARRAY_X86_FLAGS` environment variable) and `NEON` on ARM64. On x86_64 hosts, a baseline-compiled CPU check verifies `AVX2`, `FMA`, `F16C`, and `OSXSAVE` support before loading AVX2-compiled kernels and throws an actionable `UnsupportedError` if any required feature is missing.
- **OpenBLAS & LAPACK Linear Algebra**: Hardware-accelerated:
  - matrix multiplication (`matmul`),
  - Cholesky factorization (`cholesky`),
  - QR decomposition (`qr`),
  - Singular Value Decomposition (`svd`, `svdValues`),
  - symmetric/Hermitian and general eigenvalue problems (`eigh`, `eig`),
  - least-squares solvers (`lstsq`),
  - Moore-Penrose pseudo-inverse (`pinv`),
  - linear and tensor system solvers (`solve`, `tensorsolve`, `tensorinv`),
  - matrix norms, matrix rank (`matrixRank`), condition numbers (`cond`), and
  - multi-matrix chain multiplication (`multiDot`).
- **Einstein Summation (`einsum`) & Tensor Contractions**: Express arbitrary multi-dimensional tensor contractions, trace reductions, and batch matrix multiplications concisely using `einsum` (supporting explicit indices and batch ellipsis `...` notation via `EinsumSubscripts.parse('bhid,bhjd->bhij')` or `'...id,...jd->...ij'`) and generalized tensor dot products (`tensordot`), automatically dispatching to optimized BLAS GEMM paths when possible.
- **Google Highway SIMD Sorting**: Vectorized sorting and order-statistics routines (`vqsort`) leveraging Google Highway with dynamic multi-target runtime SIMD dispatch (SSE4, AVX2, AVX-512, NEON, SVE) for:
  - in-place and out-of-place `sort`,
  - indirect index sorting (`argsort`), and
  - $O(N)$ selection (`partition`, `argpartition`).
- **Spectral Transforms & DSP**:
  - Mixed-radix 1D, 2D, and N-dimensional complex and real Fast Fourier Transforms (`fft`, `ifft`, `rfft`, `irfft`, `fft2`, `fftn`) powered by `package:pocketfft`.
  - Frequency bin generators (`fftfreq`, `rfftfreq`) and zero-frequency shifting (`fftshift`, `ifftshift`).
  - 1D/2D convolution and 1D cross-correlation (`convolve`, `convolve2d`, `correlate`).
  - Spectral window functions (Hann `hanning` and Hamming `hamming`).
- **Numerical Optimization & Root Finding**:
  - Multivariate nonlinear function minimization (`minimize`) supporting quasi-Newton **L-BFGS** (`MinimizeMethod.lbfgs`) and derivative-free **Nelder-Mead** simplex (`MinimizeMethod.nelderMead`).
  - 1D scalar root finding (`rootScalar`) via **Brent's method** (`RootMethod.brentq`), Newton-Raphson, and secant iterations.
- **Spatial Distance Metrics & Polynomials**:
  - Pairwise distance matrix computations (`pdist`, `cdist`) across Euclidean, Cosine, Hamming, and Chebyshev metrics.
  - Standard polynomial fitting, evaluation, differentiation, integration, and root finding (`polyfit`, `polyval`, `polyder`, `polyint`, `roots`), plus classical orthogonal polynomial evaluation and root finding (Chebyshev, Legendre, Hermite, Laguerre).
- **Zero-Copy NumPy `.npy` & `.npz` Streaming I/O**:
  - Native binary serialization and deserialization for single arrays (`save`, `load`).
  - Multi-array ZIP archives (`savez`, `loadz` with optional compression), enabling data exchange with Python, NumPy, SciPy, and PyTorch pipelines.
- **Zero-Copy Strided Views & 64-Bit Indexing**:
  - `NDArray<T>` is backed by an off-heap C buffer with 64-bit N-dimensional `shape` and element `strides` (`int64_t`), supporting large tensors and virtual broadcasted views beyond $2^{31} - 1$ elements.
  - Provides `NDArray<Int64>` index, coordinate, and count operations (`argsort`, `argpartition`, `searchsorted`, `argmax`, `argmin`, `nonzero`, `argwhere`, `countNonzero`, `digitize`, `unravelIndex`, `ravelMultiIndex`, `indices`, `diagIndices`, `trilIndices`, `triuIndices`, `maskIndices`).
  - Operations such as multi-axis slicing (`Slice`), transposing (`.transposed`, `swapAxes`, `moveAxis`), reshaping (`reshape`), dimension expansion (`expandDims`), and squeezing (`squeeze`) return **zero-copy views** over shared C memory in $O(1)$ time.
- **Deterministic Scoped Memory Management**: Zone-based lexical resource arenas (`NDArray.scope`, `NDArray.using`) and explicit escape/cleanup helpers (`scope.escape`, `scope.defer`, `detachToParentScope`) that track and free `NDArray` allocations deterministically when a computation block completes.

---

## Installation & Build Configuration

Add `ndarray` using `dart pub add`:

```bash
dart pub add ndarray
```

Or add it to your `pubspec.yaml`:

```yaml
dependencies:
  ndarray: ^0.3.1
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

### Host Compiler Requirements (If Building from Source)

Default installs (`buildMode: fetch`) download prebuilt binaries and do not require a local C/C++ toolchain or CMake.

If desired, `buildMode: source` can be explicitly configured to build from source. Ensure the following host build tools are installed:

- **Linux**: `cmake`, `make`, and a C/C++ compiler (`gcc`/`g++` or `clang`/`clang++`).
  - *Ubuntu / Debian*: `sudo apt-get update && sudo apt-get install -y build-essential cmake gfortran`
- **macOS**: Xcode Command Line Tools (`xcode-select --install`) and `cmake` (`brew install cmake`).
- **Windows**: Visual Studio 2022 (with the *"Desktop development with C++"* workload / MSVC `cl.exe`) and `cmake` (`winget install Kitware.CMake`).

When building from source for the first time, the build hooks compile Google Highway and the `ndarray` SIMD kernels (plus OpenBLAS/LAPACK from source on Linux; macOS links the system `Accelerate` framework and Windows uses precompiled OpenBLAS), taking approximately **30 seconds to 2 minutes** depending on OS and CPU core count. Subsequent runs reuse cached artifacts in `.dart_tool/` keyed by compiler flags and source hashes.

---

## Quickstart: Idiomatic Usage & Memory Management

Because `NDArray` allocates memory on the native C heap, memory management is deterministic. Wrapping computations in `NDArray.scope(() { ... })` ensures all temporary arrays and intermediate results created inside the block are automatically freed when the scope exits.

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
> The `==` operator on `NDArray` checks **object identity**, and cannot return a boolean array due to Dart language type rules (`bool operator ==(Object other)`).
> - To compute an element-wise boolean mask comparing two arrays (equivalent to NumPy's `a == b`), use **`equal(a, b)`**, which returns an `NDArray<Boolean>`.
> - To check if two floating-point arrays are numerically equal within relative/absolute tolerances ($|a - b| \le \text{atol} + \text{rtol} \cdot |b|$, equivalent to `np.allclose(a, b)`), always use **`allClose(a, b, rtol: 1e-5, atol: 1e-8)`** or **`isClose(a, b)`**.

> [!NOTE]
> **Top-Level Math Functions and `dart:math` Namespace Conflicts**
>
> Because `package:ndarray/ndarray.dart` exports vectorized top-level math functions (`sin`, `cos`, `sqrt`, `max`, `min`, `exp`, `log`, etc.), files that also use `dart:math` for scalar operations should import `dart:math` with a prefix (`import 'dart:math' as math;`) or import `ndarray` with a prefix (`import 'package:ndarray/ndarray.dart' as np;`).

---

## Benchmarks

`package:ndarray` ships a curated, reproducible benchmark suite comparing **Dart AOT** (compiled via `dart build cli` with bundled native assets) against **PyPI NumPy** (OpenBLAS) and **SciPy** (`scipy.spatial.distance`) on identical deterministic pseudorandom inputs, using matching warmup (`200 ms` min) and `21` batch-calibrated samples (`~50 ms`/sample), reporting **median** latency with automated numerical parity checksum checks across all 25 operations.

Measured on GitHub Actions `ubuntu-latest` ([run #45396480463](https://github.com/google/scientific_dart/actions/runs/45396480463), commit `4c8ecea7`):
- **Environment:** AMD EPYC 7763 (4 logical cores), Linux `7.0.0-1020-azure`, Dart `3.14.1` AOT (`x86-64-v3` AVX2+FMA+F16C), Python `3.12.16`, NumPy `2.5.5` (`scipy-openblas` OpenBLAS `0.3.34`), SciPy `1.18.1`.
- **Single-threaded NumPy geometric mean (21 ops): `1.14x`** (`> 1.0x` means `ndarray` is faster).

| Category | Operation | `ndarray` AOT (median) | NumPy / SciPy (median) | Relative (`ndarray` vs baseline) |
| :--- | :--- | ---: | ---: | ---: |
| **Element-wise & Transcendentals** | `add(a, b, out: out) [1M f64]` | 585.6 µs | 311.4 µs | 0.53x |
|  | `add(a, b) [1M f64, allocating]` | 588.2 µs | 317.9 µs | 0.54x |
|  | `sin(x, out: out) [100k f64]` | 192.5 µs | 973.8 µs | **5.06x** |
|  | `exp(x, out: out) [100k f64]` | 178.5 µs | 441.8 µs | **2.47x** |
|  | `add(a.T, b.T, out: out) [1000x1000 f64]` | 2.41 ms | 2.11 ms | 0.88x |
| **Reductions & Statistics** | `sum(x) [1M f64]` | 109.1 µs | 197.6 µs | **1.81x** |
|  | `mean(mat, axis: 0) [1000x1000 f64]` | 155.9 µs | 164.2 µs | **1.05x** |
|  | `nansum(x) [100k f64, 10% NaN]` | 15.0 µs | 134.6 µs | **8.99x** |
| **Linear Algebra & Tensor Contractions** | `matmul(A, B) [256x256 f64, 1T]` | 777.2 µs | 762.6 µs | 0.98x |
|  | `matmul(A, B) [512x512 f32, 1T]` | 2.93 ms | 2.82 ms | 0.96x |
|  | `solve(A, b) [200x200 f64, 1T]` | 303.4 µs | 254.3 µs | 0.84x |
|  | `svd(A, fullMatrices: false) [128x128 f64, 1T]` | 8.74 ms | 2.43 ms | 0.28x |
|  | `cholesky(A) [200x200 SPD f64, 1T]` | 160.0 µs | 151.7 µs | 0.95x |
|  | `einsum('bhid,bhjd->bhij') [2x8x64x64 f32, 1T]` | 189.7 µs | 145.0 µs | 0.76x |
| **Spectral Transforms & Complex Math** | `rfft(x) [65,536 f64]` | 395.4 µs | 439.9 µs | **1.11x** |
|  | `fft2(img) [512x512 f64]` | 2.48 ms | 2.93 ms | **1.18x** |
|  | `abs(z) [100k complex128 -> f64]` | 77.9 µs | 130.9 µs | **1.68x** |
| **Sorting & Selection** | `sort(x) [100k random f64]` | 1.12 ms | 934.7 µs | 0.83x |
|  | `argsort(x) [100k random f64]` | 2.88 ms | 2.54 ms | 0.88x |
|  | `partition(x, 50000) [100k random f64]` | 188.9 µs | 183.6 µs | 0.97x |
| **Binary I/O** | `load(.npy) [1M f64, 8 MB]` | 301.5 µs | 354.7 µs | **1.18x** |
| **Spatial Distance Metrics (vs SciPy)** | `cdist(A, B, euclidean) [500x100 vs 500x100]` | 15.22 ms | 8.35 ms *(SciPy)* | 0.55x |
|  | `pdist(A, euclidean) [500x100 f64]` | 7.58 ms | 4.17 ms *(SciPy)* | 0.55x |
| **Multi-Threaded Linear Algebra** | `matmul(A, B) [1024x1024 f64, 4T]` | 48.90 ms | 23.38 ms | 0.48x |
|  | `matmul(A, B) [1024x1024 f32, 4T]` | 22.89 ms | 11.41 ms | 0.50x |

Run the suite locally from a checkout of the repository's `pkgs/ndarray` directory (using a Python environment with PyPI `numpy`, and optionally `scipy`):

```bash
dart run benchmark/run_readme_benchmark.dart --python /path/to/venv/bin/python
```

Add `--strict-blas` to fail fast if your system Python links reference Netlib BLAS instead of OpenBLAS/MKL, or `--quick` for a fast smoke check. The same suite can be re-run on GitHub Actions via the manual [`Benchmarks (Dart AOT vs NumPy/SciPy)`](https://github.com/google/scientific_dart/actions/workflows/benchmark.yml) workflow on `ubuntu-latest`.

---

## Examples

### Signal Processing — Vectorized Real FFT Denoising

This example generates a 10 Hz sine wave corrupted by Gaussian sensor noise using vectorized array operations (`NDArray.arange`, `sin`), and then:
- transforms it to the frequency domain via `rfft`,
- applies a vectorized low-pass filter using boolean mask assignment (`setByMaskScalar`), and
- reconstructs the clean time-domain signal using `irfft` (eliminating all scalar `List.generate` loops and boxed list conversions):

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

### Machine Learning — Batched Attention & PCA via `einsum` and `svd`

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
    final rawScores = einsum(
      EinsumSubscripts.parse('bhid,bhjd->bhij'),
      [q, k],
    );
    final scaledScores = rawScores * (1.0 / math.sqrt(16.0));
    print('Batched Attention Scores shape: ${scaledScores.shape}'); // [2, 4, 8, 8]

    // --- Part B: Principal Component Analysis (PCA) via SVD ---
    // Generate synthetic dataset X of shape [50 samples, 6 features]
    final x = normal([50, 6], dtype: DType.float64, seed: 42);

    // Center features by subtracting column-wise mean (keepDims: true -> [1, 6])
    final colMean = mean(x, axis: 0, keepDims: true);
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

### Scientific Optimization & NumPy `.npz` Checkpointing

This example minimizes the non-convex Rosenbrock function using quasi-Newton **L-BFGS** (`minimize(..., method: MinimizeMethod.lbfgs)`) and checkpoints the optimization parameters into a compressed NumPy `.npz` archive (`savez` / `loadz`):

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

- **[`package:resource_scope`](https://pub.dev/packages/resource_scope)**: Zone-based lexical lifetime management (`ResourceScope`) underlying `NDArray.scope` and deterministic FFI memory disposal.
- **[`package:gpuarray`](https://github.com/google/scientific_dart/tree/main/pkgs/gpuarray)** *(experimental)*: GPU-accelerated N-dimensional array computing with compute shaders and zero-copy/streaming `NDArray` interoperability.
- **[`package:ndarray_ma`](https://github.com/google/scientific_dart/tree/main/pkgs/ndarray_ma)** *(experimental)*: Masked arrays (`MaskedArray`) for `ndarray`, enabling robust computation and statistical reductions over datasets with missing or invalid entries.
- **[`package:symbolic_dart`](https://github.com/google/scientific_dart/tree/main/pkgs/symbolic_dart)** *(experimental)*: Native Computer Algebra System (CAS) symbolic mathematics with symbolic-to-numerical `ndarray` evaluation.
- **[`package:ndarray_transformer`](https://github.com/google/scientific_dart/tree/main/pkgs/ndarray_transformer)** *(experimental)*: Pure-Dart BERT / `all-MiniLM-L6-v2` sentence-embedding and semantic-search showcase loading Hugging Face `.safetensors` weights directly into `NDArray<Float32>`.

### Further Documentation

For deep dives into architecture and usage patterns, see the guides in [`doc/`](doc/):
- **[Memory Management & Scopes Guide](doc/memory_management.md)**: Detailed rules on C-heap lifetimes, views, `NDArray.scope()`, and `detachToParentScope()`.
- **[NumPy to NDArray Quickstart Guide](doc/numpy_quickstart.md)**: Side-by-side translation table and idiomatic patterns for Python/NumPy users.
- **[Shape & Stride Design Choices](doc/design_choice_shape_strides.md)**: How N-dimensional slicing, strides, and memory order are implemented.
- **[Threading & Parallelism Guide](doc/threading.md)**: Multi-threaded OpenBLAS configuration and Dart Isolate concurrency (`SendableNDArray`).

---

## License

This project is licensed under the Apache License, Version 2.0 — see the [LICENSE](https://github.com/google/scientific_dart/blob/main/pkgs/ndarray/LICENSE) file for details. Third-party notices for bundled native components are listed in [THIRD_PARTY_NOTICES](https://github.com/google/scientific_dart/blob/main/pkgs/ndarray/THIRD_PARTY_NOTICES).

## Disclaimer

This is not an official Google product.
