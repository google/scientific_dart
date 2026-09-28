# Scientific Dart

Scientific Dart is an ecosystem of packages bringing high-performance numerical, mathematical, and scientific computing to the Dart language.

The initial release centers on **`package:ndarray`**, providing a powerful N-dimensional array library inspired by NumPy, backed by optimized native BLAS, LAPACK, and FFT routines.

## Packages

The repository is organized as a Dart workspace:

- **[pkgs/ndarray](pkgs/ndarray)**: Core N-dimensional array library (`NDArray`). Features multidimensional slicing, strided views, broadcasting, universal functions (ufuncs), linear algebra solvers, statistical reductions, and random sampling.
- **[pkgs/openblas](pkgs/openblas)**: FFI bindings and Native Assets build hooks for OpenBLAS and LAPACK routines.
- **[pkgs/pocketfft](pkgs/pocketfft)**: FFI bindings and Native Assets build hooks for PocketFFT Fast Fourier Transform algorithms.
- **[pkgs/resource_scope](pkgs/resource_scope)**: Scoped resource and native memory management.

## Getting Started

### Prerequisites

* [Dart SDK](https://dart.dev/get-dart) >= 3.10.0
* A C/C++ compiler (`clang`, `gcc`, or `MSVC`) for building native assets.

### Workspace Setup

Fetch dependencies across all workspace packages:

```bash
dart pub get
```

### Running Tests

Run the test suite across packages:

```bash
dart test pkgs/*
```

### Static Analysis and Formatting

```bash
dart format --output=none --set-exit-if-changed .
dart analyze
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting patches or opening pull requests.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

## Disclaimer

This is not an official Google product.
