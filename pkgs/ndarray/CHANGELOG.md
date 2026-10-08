## 0.3.0

- **Breaking — Layered Single-Slot DType Projections**:
  - Introduced eight one-parameter projection interfaces: `RealOf<R>`, `ElementOf<E>`, `RealFloatOf<R>`, `ComplexOf<R>`, `InexactOf<R>`, `AccumulatorOf<R>`, `DoublePrecisionOf<R>`, and `DivideOf<R>` (each implementing `DTypeTag`), implemented by `DTypeSpec`.
  - Replaced the 8-positional-slot `DTypeSpec<...>` wildcard bounds across library operation signatures and operator extensions (e.g. `sin`, `sum`, `divide`, `abs`, `real`, `NDArrayDivide`, `NDArrayElements`) with the single relevant `*Of<R>` interface bound.
  - Multi-projection linear algebra operations (`slogdet`, `svd`, `eigh`) retain full `DTypeSpec` table bounds.
  - Public call sites preserve exact static type inference (NumPy promotion rules) without requiring explicit type arguments or casts.

- **Breaking — Non-Generic BitwiseDType & IntegerDType Capability Markers**:
  - Converted `BitwiseDType` and `IntegerDType` into non-generic capability markers that statically pin constant projections across integer and boolean types:
    - `BitwiseDType` pins `RealFloatOf<Float64>`, `ComplexOf<Complex128>`, `InexactOf<Float64>`, `DoublePrecisionOf<Float64>`, and `DivideOf<Float64>`.
    - `IntegerDType` extends `BitwiseDType` and additionally pins `ElementOf<int>`.
  - All eight integer tags (`Int64` through `Uint8`) implement `IntegerDType`, and `Boolean` implements `BitwiseDType`.
  - Heterogeneous collections and least upper bounds (LUBs) such as `[i64, i32]` (`List<NDArray<IntegerDType>>`) and `[i32, b]` (`List<NDArray<BitwiseDType>>`) now preserve their pinned projections statically without requiring `.asAnySpec`. For example, `sin([i64, i32].first)` now statically infers `NDArray<Float64>`.

- **Migration Guide**:
  - **Wildcard Bounds**: Replace generic helper bounds that previously spelled 8-slot wildcards such as `T extends DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag, DTypeTag>` with the single relevant projection interface, e.g. `T extends InexactOf<R>` (for transcendental and math operations), `T extends AccumulatorOf<R>` (for `sum`/`prod`), `T extends DivideOf<R>` (for division), `T extends ElementOf<E>` (for element extraction), `T extends RealFloatOf<R>`, `T extends ComplexOf<R>`, or `T extends DoublePrecisionOf<R>`.
  - **Capability Marker Type Arguments**: Remove generic type arguments from `BitwiseDType<...>` and `IntegerDType<...>`. Reference them as plain non-generic markers: `BitwiseDType` and `IntegerDType` (e.g., `NDArray<IntegerDType>` instead of `NDArray<IntegerDType<...>>`).
  - **Element Access on `NDArray<BitwiseDType>`**: Static element-access expressions (`asBitwiseDType.toList()`, `getCell`, `scalar`) on `NDArray<BitwiseDType>` now infer `List<dynamic>` / `dynamic` rather than `List<Object>` / `Object`. This occurs because `BitwiseDType` encompasses both integer tags (`ElementOf<int>`) and `Boolean` (`ElementOf<bool>`), and Dart interface type parameters are invariant. When integer element types are needed, use `NDArray<IntegerDType>` or `.asIntegerDType` (which provides `ElementOf<int>`).

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
