# OpenBLAS for Dart

A Dart native library providing high-performance linear algebra operations through bindings to OpenBLAS. This package allows you to leverage highly optimized, multi-threaded BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package) routines directly from your Dart code.

## Thread Safety

This package is designed to be safe for use across multiple Dart isolates. The build hook explicitly enables threading support when compiling OpenBLAS, allowing concurrent calls from different isolates to share the underlying native library safely.
