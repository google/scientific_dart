## 0.2.0

- **64-Bit Array Dimensions, Strides, & Native Kernels**: Removed the 32-bit (`2^31 - 1`) element count, shape dimension, and stride ceiling across `NDArray` creation, slicing, broadcasting, `.npy`/`.npz` I/O, and native C/C++ kernels (`int64_t`), with division-based 64-bit signed integer overflow checks.
- **Breaking — `NDArray<Int64>` Index & Count Operations**: Widened `argsort`, `argpartition`, `searchsorted`, `argmax`, `argmin`, `nonzero`, `flatnonzero`, `argwhere`, `count_nonzero`, and `digitize` (along with `NDArray.argsort`, `NDArray.argmax`, `NDArray.argmin`, and 1-argument `where`) from `NDArray<Int32>` to `NDArray<Int64>` to match 64-bit NumPy `intp` (`argsortAs`, `argpartitionAs`, `searchsortedAs`, and `digitizeAs` remain available for explicit `DType.int32` outputs).
- **OpenBLAS / LAPACK `blasint` Boundary Guards**: Added explicit LP64 32-bit `blasint` boundary checks across linear algebra and tensor contraction routines, throwing an `UnsupportedError` when matrix dimensions or leading strides exceed `2^31 - 1`.

## 0.1.0

- Initial release with N-dimensional arrays, slicing, broadcasting, BLAS/LAPACK linear algebra, and FFT.
