## 0.2.1

- **Mixed-DType `*As` Binary Operations**: Allowed mixed input dtypes (`<Ta, Tb, R>`) in `addAs`, `subtractAs`, `multiplyAs`, and `divideAs` when the target `DType<R>` is explicitly specified, dispatching directly to single-pass mixed-dtype C++/SIMD kernels when the kernel output matches `dtype`.
- **Expanded `*As` & Binary Operation Coverage**: Added `floorDivideAs`, `remainderAs`, `modAs`, `fmodAs`, `divmodAs`, `powerAs`, `floatPower`, `floatPowerAs`, `minimum`, `minimumAs`, `maximum`, `maximumAs`, `fmin`, `fminAs`, `fmax`, `fmaxAs`, `matmulAs`, `dot`, `dotAs`, `tensordotAs`, `innerAs`, `vdotAs`, `kronAs`, `outerAs`, `crossAs`, `atan2As`, `hypotAs`, `logaddexpAs`, `logaddexp2As`, `copysignAs`, `heavisideAs`, `gcdAs`, `lcmAs`, `bitwiseAndAs`, `bitwiseOrAs`, `bitwiseXorAs`, `leftShiftAs`, and `rightShiftAs`.
- **Static Same-DType Type Parameter Tightening**: Tightened non-`*As` same-dtype binary operations (`divide`, `logaddexp`, `logaddexp2`, `atan2`, `hypot`, `logicalAnd`, `logicalOr`, and `logicalXor`) so both operands share a single input type parameter `T` statically, matching their runtime same-dtype requirement.

## 0.2.0

- **64-Bit Array Dimensions, Strides, & Native Kernels**: Removed the 32-bit (`2^31 - 1`) element count, shape dimension, and stride ceiling across `NDArray` creation, slicing, broadcasting, `.npy`/`.npz` I/O, FFT (`package:pocketfft` `0.2.0`), and native C/C++ kernels (`int64_t`), with division-based 64-bit signed integer overflow checks.
- **Breaking — `NDArray<Int64>` Index, Count, & Choice Operations**: Widened `argsort`, `argpartition`, `searchsorted`, `argmax`, `argmin`, `nonzero`, `flatnonzero`, `argwhere`, `count_nonzero`, and `digitize` (along with `NDArray.argsort`, `NDArray.argmax`, `NDArray.argmin`, and 1-argument `where`) from `NDArray<Int32>` to `NDArray<Int64>` to match 64-bit NumPy `intp` (`argsortAs`, `argpartitionAs`, `searchsortedAs`, and `digitizeAs` remain available for explicit `DType.int32` outputs). Updated `select` and `multinomial` to default to `DType.int64` instead of `DType.int32`, and aligned `choose` and `select` with NEP 50 weak scalar promotion rules when mixing Dart `int` scalars with narrow integer arrays.
- **64-Bit Index Generation & Coordinate Utilities**: Added `unravel_index` (`unravelIndex`), `ravel_multi_index` (`ravelMultiIndex`), `indices`, `sparse_indices` (`sparseIndices`), `diag_indices` (`diagIndices`), `diag_indices_from` (`diagIndicesFrom`), `tril_indices` (`trilIndices`), `tril_indices_from` (`trilIndicesFrom`), `triu_indices` (`triuIndices`), `triu_indices_from` (`triuIndicesFrom`), `mask_indices` (`maskIndices`), and `IndexOrder`, all returning `NDArray<Int64>` coordinates.
- **OpenBLAS / LAPACK `blasint` Boundary Guards**: Added explicit LP64 32-bit `blasint` boundary checks across linear algebra and tensor contraction routines, throwing an `UnsupportedError` when matrix dimensions or leading strides exceed `2^31 - 1`.

## 0.1.0

- Initial release with N-dimensional arrays, slicing, broadcasting, BLAS/LAPACK linear algebra, and FFT.
