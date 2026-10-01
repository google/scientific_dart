# `package:ndarray` — Remaining Issues & Post-Review Backlog

This file tracks all verified remaining work items identified during the production-readiness review (see [`PRODUCTION_READINESS_REVIEW.md`](PRODUCTION_READINESS_REVIEW.md)). Every item below has been verified against the current source tree.

---

## P0: External Pre-Publish Blockers (Before `dart pub publish`)

- [ ] **P0-1: Publish sibling workspace packages to `pub.dev` in topological order**
  - **Location**: [`pubspec.yaml:37-44`](pubspec.yaml#L37-L44)
  - **Details**: `ndarray` depends on workspace siblings `resource_scope: ^0.1.0`, `openblas: ^0.1.0`, and `pocketfft: ^0.2.0`. Publish `resource_scope` first, then `openblas` and `pocketfft`, before publishing `ndarray`.

- [ ] **P0-2: Build and upload prebuilt native binaries (`artifacts-v0.2.0`) & regenerate hashes**
  - **Location**: [`hook/build.dart:44-75`](hook/build.dart#L44-L75), [`lib/src/hook_helpers/hashes.dart:14-41`](lib/src/hook_helpers/hashes.dart#L14-L41)
  - **Details**: Native C++ sources (`hook/custom_ufuncs.cpp`, `hook/custom_sorting.cpp`, `hook/custom_indexing.cpp`, `hook/npz_io.cpp`) were modified during the review remediation. Before cutting the release:
    1. Build `libndarray_c` across all release targets (`linux-x64`, `linux-arm64`, `macos-arm64`, `macos-x64`, `windows-x64`) via CI.
    2. Upload the archives to the GitHub Release `artifacts-v0.2.0`.
    3. Run `dart run tool/regenerate_hashes.dart` to update `expectedSourceHash` and `prebuiltHashes` in `lib/src/hook_helpers/hashes.dart`.

---

## P1: Performance & Native C/SIMD Kernel Gaps (All Completed)

- [x] **P1-1: Broaden Highway `VQSort` SIMD vectorization in `ndarray_unique`**
  - **Location**: [`hook/custom_sorting.cpp:2570-2960`](hook/custom_sorting.cpp#L2570-L2960)
  - **Resolution**: Added 256-bucket counting sort for `Int8` (`unique_int8_fast` using `^ 0x80` sign-bit flip), Highway `VQSort` kernels for `Uint16`, `Uint32`, and `Uint64` (`unique_uint16_fast`, `unique_uint32_fast`, `unique_uint64_fast`), and NaN-partitioned Highway `VQSort` kernels for `Float32` and `Float64` (`unique_float_fast`, `unique_double_fast` with single-NaN compaction), and wired all 10 into `ndarray_unique`. Enforced by `test/meta/codebase_invariants_test.dart` and verified across all 14 `DType`s in `test/meta/operation_contracts_test.dart`.

- [x] **P1-2: Native C SIMD kernels for binary elementwise `minimum`, `maximum`, `fmin`, and `fmax`**
  - **Location**: [`lib/src/operations/math/ufunc_methods.dart:444-550`](lib/src/operations/math/ufunc_methods.dart#L444-L550), [`hook/custom_ufuncs.cpp`](hook/custom_ufuncs.cpp)
  - **Resolution**: Implemented contiguous Highway SIMD (`v_binary_minmax`) and strided (`s_binary_minmax`) C kernels in `hook/custom_ufuncs.cpp` across all 14 `DType`s (including IEEE-754 `-0.0` vs `+0.0` signbit handling, NaN propagation vs ignoring, and complex lexicographical ordering), fixed `apply_at_op<cpx_t>` / `apply_at_op<cpx_f_t>` for `OP_MINIMUM`/`OP_MAXIMUM`/`OP_FMIN`/`OP_FMAX`, and wired `_nativeMinMax` into `binaryUfunc`.

- [x] **P1-3: Native C kernels for `accumulateUfunc` and `outerUfunc`**
  - **Location**: [`lib/src/operations/math/ufunc_methods.dart:1627-1960`](lib/src/operations/math/ufunc_methods.dart#L1627-L1960) (`accumulateUfunc`), [`lines 2850-2900`](lib/src/operations/math/ufunc_methods.dart#L2850-L2900) (`outerUfunc`)
  - **Resolution**: Removed redundant Dart pointer loops in `accumulateUfunc` so all supported types (including `Int16` and `Uint8`) dispatch directly to native C `s_cumsum_*` / `s_cumprod_*` / `s_cummin_*` / `s_cummax_*`, and refactored `outerUfunc` to delegate directly to `binaryUfunc` on reshaped views using native strided/broadcast C kernels.

- [x] **P1-4: Native C kernels for remaining Dart-loop reductions in `stats.dart`**
  - **Location**: [`lib/src/operations/stats.dart`](lib/src/operations/stats.dart)
  - **Resolution**: Delegated `nansum` on integer/boolean arrays directly to `sum<R>` (native C `r_sum_*` / `s_sum_*`), cast `Float16`/`BFloat16` to `Float32` for native `r_nansum_float` / `s_nansum_float`, wired `all` and `any` (both full and axis reductions across all `DType`s) to native C `v_to_bool_*` / `s_to_bool_*` + `r_all_bool` / `r_any_bool` / `s_logical_and_red` / `s_logical_or_red`, and replaced boxed `getCell`/`ndenumerate` loops in `nanvar`, `nansum`, and complex `min`/`max` with native C and fast typed FFI pointer loops.

- [x] **P1-5: Eliminate `.toList()` / `getCellFlat` round-trips in indexing (`ndarray.dart`)**
  - **Location**: [`lib/src/ndarray.dart`](lib/src/ndarray.dart)
  - **Resolution**: Replaced `.asTypedList(count).toList()` with unboxed `Int64List.fromList(...)` in `_sliceAssignImpl` and `slice`, added `_extractInt64Indices` using contiguous `Int64` views / `castNDArray<Int64>` instead of per-element `getCellFlat` loops in `_toSelector`, `operator []`, and `operator []=`, and removed redundant `.toList()` calls after `.cast<int>()`.

---

## P2: API Ergonomics, Strong Typing, & Documentation Consistency (All Completed)

- [x] **P2-1: Split or strongly type `unique` instead of returning `dynamic` based on boolean flags**
  - **Location**: [`lib/src/operations/set_operations.dart:32-310`](lib/src/operations/set_operations.dart#L32-L310)
  - **Resolution**: Changed `unique<T extends DTypeTag>(NDArray<T> ar, {NDArray<T>? out})` to return `NDArray<T>` directly, and added strongly-typed named-record functions `uniqueWithIndex<T>`, `uniqueWithInverse<T>`, `uniqueWithCounts<T>`, and `uniqueAll<T>`.

- [x] **P2-2: Convert positional record returns in `divmod` and `ndenumerate` to named records**
  - **Location**: [`lib/src/operations/math/arithmetic.dart:3557`](lib/src/operations/math/arithmetic.dart#L3557) (`divmod`), [`lib/src/operations/math/utility.dart:65`](lib/src/operations/math/utility.dart#L65) (`ndenumerate`)
  - **Resolution**: Updated `divmod<T>` to return `({NDArray<T> quotient, NDArray<T> remainder})` and `ndenumerate<T>` to yield `({List<int> coordinate, Object value})`. Enforced across all exported top-level functions and `NDArray` methods in `test/meta/codebase_invariants_test.dart`.

- [x] **P2-3: Tighten remaining `dynamic` / un-generic `NDArray` signatures**
  - **Location**: [`lib/src/ndarray.dart`](lib/src/ndarray.dart), [`lib/src/operations/`](lib/src/operations/)
  - **Resolution**: Replaced all remaining `dynamic` and raw `NDArray` parameter/return types across the public API (`squeeze`, `moveaxis`, `flip`, `partition`, `fftshift`, `ifftshift`, `nan_to_num`, `ndenumerate`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `meshgrid`, `histogramdd`, `diff`, `ediff1d`, `gradient`, `linspaceWithStep`). Enforced via resolved `DartType` AST checks in `test/meta/codebase_invariants_test.dart`.

- [x] **P2-4: Align `nansum` integer/boolean accumulation dtype with `sum`**
  - **Location**: [`lib/src/operations/stats.dart:6099-6140`](lib/src/operations/stats.dart#L6099-L6140)
  - **Resolution**: `nansum` now widens narrow signed integers (`Int8`, `Int16`, `Int32`) and `Boolean` to `Int64` and narrow unsigned integers (`Uint8`, `Uint16`, `Uint32`) to `Uint64` via `_defaultAccumDType` and delegates integer/boolean arrays to `sum<R>`. Verified across all 14 `DType`s in `test/meta/operation_contracts_test.dart`.

- [x] **P2-5: Standardize remaining out-of-bounds `axis` errors to `RangeError.range`**
  - **Location**: [`lib/src/operations/dsp.dart`](lib/src/operations/dsp.dart), [`lib/src/operations/calculus.dart`](lib/src/operations/calculus.dart), [`lib/src/operations/linalg.dart`](lib/src/operations/linalg.dart), [`lib/src/operations/manipulation.dart`](lib/src/operations/manipulation.dart), [`lib/src/operations/spacers.dart`](lib/src/operations/spacers.dart)
  - **Resolution**: Standardized out-of-bounds `axis` checks to throw `RangeError.range`. Verified by cross-cutting behavioral contract tests across 40+ axis-accepting operations in `test/meta/operation_contracts_test.dart`.

- [x] **P2-6: Expand `{@example ...}` tag coverage to the remaining 7 operation modules**
  - **Location**: `binning.dart`, `broadcasting.dart`, `calculus.dart`, `dsp.dart`, `indexing.dart`, `repeating_tiling.dart`, `set_operations.dart`, and `example/*_example.dart`
  - **Resolution**: Converted all inline ```` ```dart ```` blocks in these 7 modules to `{@example /example/... lang=dart}` tags backed by standalone analyzer-clean files under `example/`. Enforced in `test/meta/codebase_invariants_test.dart`.

---

## P3: Long-Term Format & I/O Enhancements

- [ ] **P3-1: ZIP64 large archive support (> 4 GiB) in `npz_io.cpp`**
  - **Location**: [`hook/npz_io.cpp:201-425`](hook/npz_io.cpp#L201-L425)
  - **Details**: `npz_save_stored` and `npz_save_compressed` write 32-bit ZIP headers and return `-4` (translated to `UnsupportedError` in Dart) when cumulative offsets or member sizes exceed `0xFFFFFFFF` (4 GiB). Implement ZIP64 extended information extra fields (`0x0001`), ZIP64 End of Central Directory record (`0x06064b50`), and ZIP64 End of Central Directory Locator (`0x07064b50`) to support reading and writing `.npz` archives larger than 4 GiB.
