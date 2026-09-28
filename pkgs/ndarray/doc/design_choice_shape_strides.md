# Design Choice: Shape and Strides Storage

This document explains why the `ndarray` package stores shape and strides as Dart lists rather than in C memory, despite calling C kernels for performance.

## Context
NumPy, the inspiration for this library, stores shape and strides in C memory. This is natural because NumPy is a C extension for Python, where Python object access is slow.

## Why Dart Lists?
1.  **Fast Dart Access**: Dart is a compiled language with fast list access. Accessing native memory via FFI pointers is slower than accessing Dart lists due to boundary-crossing overhead. Since shape and strides are frequently inspected in Dart (for validation, broadcasting, and slicing), keeping them in Dart is faster for metadata operations.
2.  **Lightweight Views without C Allocations**: While a root `NDArray` could pack its `shape`, `strides`, and data buffer into a single C allocation freed together, **views** (slices, transposes, reshapes) have their own distinct `shape` and `strides` while sharing the root array's data pointer. Storing `shape` and `strides` as Dart lists means views own no C-heap memory of their own, need no `NativeFinalizer` or manual disposal, and are collected directly by the Dart garbage collector like any ordinary Dart object.

## FFI Call Optimization
To avoid allocating C memory when passing `shape` and `strides` to C kernels, FFI calls copy them into `ScratchArena` (a reusable isolate-local bump buffer).
