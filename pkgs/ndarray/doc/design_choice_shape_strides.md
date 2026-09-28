# Design Choice: Shape and Strides Storage

This document explains why the `ndarray` package stores shape and strides as Dart lists rather than in C memory, despite calling C kernels for performance.

## Context
NumPy, the inspiration for this library, stores shape and strides in C memory. This is natural because NumPy is a C extension for Python, where Python object access is slow.

## Why Dart Lists?
1.  **Fast Dart Access**: Dart is a compiled language with fast list access. Accessing native memory via FFI pointers is slower than accessing Dart lists due to the boundary crossing overhead. Since shape and strides are frequently inspected in Dart (for validation, broadcasting, and slicing), keeping them in Dart is more efficient for most operations.
2.  **Memory Management**: Dart handles garbage collection for lists. If stored in C, we would need manual memory management for shape and strides arrays, complicating the lifecycle of `NDArray` and its views.
3.  **View Independence**: When creating a view (slice), it has its own shape and strides but shares data. Storing them in Dart makes creating views lightweight without calling `malloc` for new shape/strides arrays.

## FFI Call Optimization
To avoid the overhead of copying Dart lists to C memory on every FFI call, we use a reusable isolate-local buffer. This eliminates allocation overhead while keeping the benefits of Dart-side storage.

## Conclusion
Our current approach (Dart lists + global buffer for FFI) is a good trade-off for Dart/C interop. It prioritizes fast access in Dart while maintaining performance for C kernel execution.
