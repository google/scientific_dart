# Scientific Dart

<img src="doc/images/mascot.png" align="right" width="220" alt="Scientific Dart mascot">

Scientific Dart is an ecosystem of packages bringing high-performance numerical, mathematical, and scientific computing to the Dart language.

The initial release centers on **`package:ndarray`**, providing a powerful N-dimensional array library inspired by NumPy, backed by optimized native SIMD, BLAS, LAPACK, and FFT routines.

## Packages

The repository is organized as a Dart workspace configured under [`pubspec.yaml`](pubspec.yaml):

### Core Packages

- **[`pkgs/ndarray`](pkgs/ndarray)**: The core scientific N-dimensional array and tensor computing library (`NDArray`). Features native C/C++17 & Google Highway SIMD kernels, OpenBLAS/LAPACK linear algebra (`matmul`, `svd`, `qr`, `cholesky`, `eigh`/`eig`, `lstsq`, `pinv`, `solve`), Einstein summation (`einsum`) & `tensordot`, Google Highway SIMD sorting (`sort`, `argsort`, `partition`, `argpartition`), spectral transforms & DSP (`fft`, `rfft`, `convolve`, `correlate`, windows), numerical optimization & root finding (`minimize`, `root_scalar`), spatial distance metrics (`pdist`, `cdist`), orthogonal polynomials, zero-copy NumPy `.npy`/`.npz` streaming I/O, zero-copy strided views, and deterministic scoped memory management (`NDArray.scope`).
- **[`pkgs/openblas`](pkgs/openblas)**: Low-level FFI bindings and Native Assets build hooks wrapping OpenBLAS CBLAS and LAPACK routines across Linux, macOS, and Windows.
- **[`pkgs/pocketfft`](pkgs/pocketfft)**: Native AOT FFI bindings and Native Assets build hooks around PocketFFT/KissFFT mixed-radix discrete Fourier transform plans for fast 1D and multidimensional complex and real FFTs.
- **[`pkgs/resource_scope`](pkgs/resource_scope)**: Zone-based automatic scoped resource management (`ResourceScope`, `NDArray.scope`) for deterministic FFI and native C-heap memory disposal in Dart.

### Experimental Packages & Applications

> [!WARNING]
> The packages below are **experimental** and under active development. Their APIs may change significantly or be removed before reaching a stable release.

- **[`pkgs/gpuarray`](pkgs/gpuarray)** *(experimental)*: GPU-accelerated N-dimensional array computing with compute shaders and zero-copy/streaming interoperability with `NDArray`.
- **[`pkgs/ndarray_ma`](pkgs/ndarray_ma)** *(experimental)*: Masked arrays (`MaskedArray`) for `ndarray`, enabling element-wise operations, reductions, and statistical analysis over datasets with missing, invalid, or masked entries.
- **[`pkgs/symbolic_dart`](pkgs/symbolic_dart)** *(experimental)*: Symbolic mathematics and Computer Algebra System (CAS) library for Dart powered by native C/C++ bindings (SymEngine & FLINT), with symbolic-to-numerical `ndarray` evaluation.
- **[`pkgs/scientific_dart_analysis_plugin`](pkgs/scientific_dart_analysis_plugin)** *(experimental)*: Analyzer plugin providing memory-safety, view-lifecycle, `DType`, and performance lints and quick fixes for `scientific_dart` consumers.
- **[`pkgs/notebook`](pkgs/notebook)** *(experimental)*: Interactive notebook and REPL interface for Dart, `ndarray`, and `symbolic_dart`.
- **[`pkgs/code_editor`](pkgs/code_editor)** *(experimental)*: Multi-backend code editor component supporting syntax highlighting and interactive evaluation for the notebook environment.
- **[`pkgs/guitar_tuner`](pkgs/guitar_tuner)** *(experimental)*: Real-time CLI guitar tuner demonstrating live audio capture (ALSA) and spectral pitch detection using `ndarray`.

---

## Getting Started

### 1. Prerequisites
- [Dart SDK](https://dart.dev/get-dart) `^3.10.0`
- A C/C++ compiler (`clang`, `gcc`, or `MSVC`) and `cmake` when building native assets from source.

### 2. Workspace Setup
Fetch dependencies across all workspace packages at once from the repository root:
```bash
dart pub get
```

### 3. Code Formatting & Static Analysis
Ensure formatting and static analysis pass with zero warnings or errors before submitting changes:
```bash
dart format --output=none --set-exit-if-changed .
dart analyze
```

### 4. Running Tests
To run unit tests across a specific package (e.g., `pkgs/ndarray`):
```bash
dart test pkgs/ndarray
```

### 5. Generating Coverage Reports
To measure test coverage metrics inside `ndarray`, navigate to `pkgs/ndarray` and run:
```bash
dart tool/generate_coverage.dart
```

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting patches or opening pull requests.

## License

This project is licensed under the **[Apache License, Version 2.0](LICENSE)**.

## Disclaimer

This is not an official Google product.
