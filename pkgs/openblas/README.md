# OpenBLAS for Dart

<img src="doc/images/mascot.png" align="right" width="220" alt="openblas mascot">

A Dart native library providing high-performance linear algebra operations through bindings to OpenBLAS. This package allows you to leverage highly optimized, multi-threaded BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package) routines directly from your Dart code.

## Thread Safety

This package is designed to be safe for use across multiple Dart isolates. The build hook explicitly enables threading support when compiling OpenBLAS, allowing concurrent calls from different isolates to share the underlying native library safely.

## Build Configuration (`hooks.user_defines`)

By default (`buildMode: fetch`), `package:openblas` downloads prebuilt shared libraries (`libopenblas` and `libopenblas_extensions`) verified against pinned SHA-256 digests and embedded source hashes, automatically falling back to `buildMode: source` if prebuilt binaries are unavailable or stale.

You can customize the build mode or enable C sanitizers/coverage for `openblas_extensions.c` in your root `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    openblas:
      # 'fetch' (default), 'source', or 'local'
      buildMode: source
      # Optional path to an existing OpenBLAS source checkout when buildMode is 'source':
      # checkoutPath: /path/to/OpenBLAS
      # Optional C sanitizers ('address', 'undefined', 'address,undefined') for openblas_extensions.c:
      # sanitize: address,undefined
      # Optional coverage instrumentation (true / false) for openblas_extensions.c:
      # coverage: false
      # Required paths when buildMode is 'local':
      # localPath: /path/to/dist/openblas-linux-x64.so
      # localExtensionsPath: /path/to/dist/openblas_extensions-linux-x64.so
```

