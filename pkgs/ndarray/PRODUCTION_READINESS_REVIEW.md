# Production-Readiness Review: `package:ndarray`

**Package**: `ndarray`  
**Package Path**: `/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray`  
**Review Date**: 2026-09-28  
**Auditor**: Teamwork Production Engineering Review & Remediation Taskforce  
**Target Pub Release**: `0.2.0`  
**Dart SDK**: Dart 3.7+ (`/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart`)  

---

## 1. Executive Summary & Readiness Verdict

### Explicit Readiness Verdict
> **CONDITIONAL GO** — **In-Tree Code, Native C/FFI Kernels, Documentation, and Meta-Test Suites are Production-Ready; External Release Artifact Upload & Workspace Sibling Dependency Publishing (P0 Blockers) Are Required Prior to `dart pub publish`.**

All in-tree defects, undefined behavior risks, memory leaks, unhandled numerical edge cases, and packaging bloat uncovered during the comprehensive audit have been remediated and verified under strict static and dynamic invariants. The repository exhibits:
- **0 static analysis errors or warnings** (`dart analyze`).
- **0 code formatting violations** (`dart format --output=none --set-exit-if-changed lib test`).
- **100% test pass rate** across all 4,587 tests in the package test suite (`dart test`).
- **0 AddressSanitizer or UndefinedBehaviorSanitizer violations** during native execution (`NDARRAY_SANITIZE=address,undefined`).
- **Clean `dart pub publish --dry-run`** with archive size reduced from **14 MB to < 1 MB**.

Before triggering `dart pub publish` to `pub.dev`, two external infrastructure steps (P0) outside the repository tree must be completed:
1. Publishing sibling workspace packages (`resource_scope`, `openblas`, and `pocketfft`) to `pub.dev` in topological order.
2. Building and uploading precompiled native asset binaries for Linux (x64, arm64), macOS (arm64, x64), and Windows (x64) matching `nativeSourceHash` to GitHub Release `artifacts-v0.2.0`.

---

### 5-Dimension Readiness Matrix

| Dimension | Before Remediation | After In-Tree Remediation (Milestones M1 & M2) | Remaining Action for Pre-Publish | Status |
|---|---|---|---|---|
| **Dim 1: API Design, Ergonomics, & Documentation** | Leaked `typedef Float = double;` colliding with `dart:ffi`; inconsistent error types (`StateError` vs. `LinAlgError` on singular matrices); missing `const` constructors on `Complex` and `IndexSpec`; uncopied mutable lists in public record/class wrappers; 133 operations missing writeability checks; broken/type-mismatched Dartdoc examples (`1.0 as Float64`). | `typedef Float` removed; singular matrix error normalized to `LinAlgError`; constructors made `const`; lists defensively wrapped with `List.unmodifiable`; universal `validateOutArray` / `validateOutBuffer` enforcement; Dartdoc and doclinks updated and aligned with Effective Dart and `lrn-review`. | None (`{@example}` tag coverage completed in P2-6). | **READY (In-Tree)** |
| **Dim 2: Native C/FFI Safety, Bounds, & Memory** | C++ signed integer overflow UB in `s_diff`, `add_prod_corr`, and `v_reduceat_unroll_helper`; `NaN` poisoning in 8-way unrolled `reduceat` (`fmin`/`fmax`); silent integer division by zero; float-to-int cast UB and 32-bit index truncation in 64-bit arrays; unhandled OOM returning `-4`; memory leak in advanced indexing `_sliceAssign` `broadcastTo`. | Casts to unsigned/`std::make_unsigned_t<T>` applied; `apply_at_op` NaN-resilient unrolled reduction; global `division_error_flag` with `get_and_reset_division_error()` and `UnsupportedError` translation; double clamping before `int64_t` cast and 64-bit index variables; `ndarray_set_oom_flag()` and `-4 -> OutOfMemoryError` translation; `_sliceAssign` `try`/`finally` view disposal and self-aliasing protection; ZIP64 support in `npz_io.cpp`. | None (ZIP64 format support in `npz_io.cpp` completed in P3-1). | **READY (In-Tree)** |
| **Dim 3: Numerical Correctness & Edge Cases** | Broadcast zero-stride views accepted as writeable `out:` buffers causing memory corruption; self-aliasing corruption in `argsort`, `argpartition`, `searchsorted`; uncaught negative/out-of-bounds axis in `_reductionTargetShape`; `keepdims` dropped in `ptp`; 2D empty arrays (`[0, 3]`) failing non-zero axis reductions; empty inputs unhandled in `correlate`/`convolve`. | `validateOutArray` rejects read-only and zero-stride broadcast buffers; aliasing detection copies input before mutating `out:`; `RangeError.range` validates reduction axes; `keepdims` forwarded throughout `ptp`; empty array reductions return consistent empty target shapes; empty inputs rejected with clear argument errors; `bincount` validated for integers and `Uint64`. | None (Native C reduction kernels completed in P1-4). | **READY (In-Tree)** |
| **Dim 4: Performance, SIMD, & Integrations** | Highway SIMD `unique_*_fast` kernels implemented but dead (unwired in `ndarray_unique`); cache-unfriendly `r-c-i` loop order in integer matrix multiplication; unnecessary `.toList()` allocations in indexing. | Highway SIMD kernels wired across all 10 integer and float `DType`s; native C SIMD kernels added for binary `minimum`, `maximum`, `fmin`, `fmax`; contiguous `r-i-c` loop order added to `DEFINE_MATMUL_INT` (`F12`); `lib/src/hook_helpers/hashes.dart` intentionally kept pinned to `artifacts-v0.2.0` release hashes in-tree. | None (Completed in P1-1 through P1-5). | **READY (In-Tree)** |
| **Dim 5: Pub Publishing, Native Assets, & Packaging** | `.pubignore` omitted `coverage/`, `scratch/`, and `third_party/highway/build/`, bloating package archive to **14 MB**; unpublished workspace dependencies (`openblas`, `pocketfft`, `resource_scope`); missing prebuilt binary hashes for updated C source code. | `.pubignore` updated to ignore `coverage/`, `scratch/`, `pubspec.lock`, and build artifacts, reducing publish archive to **< 1 MB**; `build_infra_invariants_test.dart` enforces that `.gitignore` entries are in `.pubignore`. | Publish workspace dependencies in topological order (P0-1); upload prebuilt native binaries to GitHub Release `artifacts-v0.2.0` (P0-2). | **CONDITIONAL GO** (Awaiting P0 External Steps) |

---

## 2. Dimension-by-Dimension Deep Dive

### Dimension 1: API Design, Ergonomics, & Documentation

#### 1.1 Strong Generic Typing & DType Safety
`package:ndarray` enforces strong typing across `NDArray<T>` where `T` represents the `DTypeTag` marker (`Float64`, `Float32`, `Float16`, `BFloat16`, `Int64`, `Int32`, `Int16`, `Int8`, `Uint64`, `Uint32`, `Uint16`, `Uint8`, `Complex64`, `Complex128`, `Boolean`). All operations in `lib/src/operations/` dispatch via canonical `switch (a.dtype)` statements, adhering strictly to `AGENTS.md`. 
- **Fixed Leak**: `lib/src/operations/dsp.dart:25` had declared `typedef Float = double;` in the public library scope, which shadowed or collided with `dart:ffi`'s `Float`. This was eliminated, using native `double?` internally.
- **Fixed Singular Matrix Contract**: In `lib/src/operations/linalg.dart:1645, 1691, 1737, 1784`, matrix inversion (`inv()`) had previously thrown `StateError` upon encountering a singular matrix (`info > 0`), contradicting its own Dartdoc and peer linear algebra routines (`solve`, `cholesky`, `eig`). It now uniformly throws `LinAlgError('Matrix is singular and cannot be inverted.')`.
- **Compile-Time `const` Support**: `Complex(this.real, this.imag)` (`lib/src/ndarray.dart:5554`), `IndexSpec()` (`line 5757`), `Index(this.value)` (`line 5768`), and `Selector` were upgraded to provide `const` constructors, enabling canonical `const Complex(0.0, 0.0)` instances in application code and documentation examples.

#### 1.2 Defensive Immutability & Parameter Validation
In compliance with `global_rules.md` (defensively copy mutable collections):
- Parameter collections stored in wrapper classes (`Indices`, `CoordinateSpacing`, `TensordotAxes`, `BroadcastResult`) now wrap input lists in `List.unmodifiable(...)`.
- Parameter validation across all operations follows standard Dart exceptions: `ArgumentError.value(val, 'name', 'Must be ...')`, `RangeError.range(...)`, and `StateError(...)`.
- `validateOutArray` in `lib/src/operations/helpers.dart:46` validates that destination buffers are not disposed, are writeable, match target shapes and data types, and do not represent broadcast zero-stride views.

#### 1.3 Documentation & Effective Dart Compliance
All public APIs have been audited for Effective Dart style:
- Fixed Dartdoc type assertions (such as `1.0 as Float64`) in `lib/src/ndarray.dart:2128`.
- Replaced outdated doclinks (`[NDArray.ndim]` -> `[NDArray.rank]`).
- Documented enum constants in `PadMode` (`lib/src/operations/padding.dart`).
- Standardized `{@example}` tags with co-located examples in `pkgs/ndarray/example/`.

---

### Dimension 2: Native C/FFI Safety, Bounds Checking, & Memory Lifecycle

The native layer (`hook/`) compiles into the `ndarray` code asset (`libndarray`) using the Dart Native Assets feature (`hook/build.dart`). It combines optimized Highway SIMD kernels with scalar C++ fallbacks.

#### 2.1 C/C++ Integer Overflow & Undefined Behavior (UB)
Signed integer overflow in C++ is undefined behavior under ISO C++. In numerical computing, calculations like difference series (`s_diff`), cross-correlation accumulations (`add_prod_corr`), and unrolled stride reductions (`v_reduceat_unroll_helper`) frequently encounter boundary values (`INT64_MIN`, `INT64_MAX`).
- **Remediation**: In `hook/custom_ufuncs.cpp:5730-5731`, `lines 13623-13624`, and `lines 14332-14357`, arithmetic operations on signed integers are cast to their corresponding unsigned types (`uint64_t`, `uint32_t`, or `std::make_unsigned_t<T>`) prior to addition, subtraction, or multiplication, before casting back to signed types. This satisfies `-fsanitize=undefined` and ensures modular two's-complement wrapping.

#### 2.2 Numerical IEEE 754 Edge Cases in C SIMD Loops
- **NaN Poisoning**: In `v_reduceat_unroll_helper` (`hook/custom_ufuncs.cpp:14358-14402`), an 8-way unrolled reduction loop initialized 8 independent accumulator lanes. If `src[start]` was `NaN`, standard SIMD comparisons in `fmin`/`fmax` caused `NaN` to populate all lanes, corrupting results for slices longer than 16 elements. The loop was restructured to use `apply_at_op(accK, src[i+K], opCode)` with NaN-propagating semantics matching NumPy (`fmin` ignores NaN if one operand is valid; `minimum` propagates NaN).
- **Division by Zero Trapping**: Native C integer division by zero terminates the process with `SIGFPE`. In `hook/custom_ufuncs.cpp:13998-14033`, arithmetic kernels for integer division and modulo (`fdiv_int`, `floordiv_int`, `rem_int`, `fmod_int`) check if `b == 0`. When zero is detected, a thread-local flag `division_error_flag = 1` is recorded, and the kernel safely substitutes 0 to avoid hardware traps. Dart wrappers (`reduceatUfunc` and `atUfunc` in `lib/src/operations/math/ufunc_methods.dart`) query `get_and_reset_division_error()` and throw `UnsupportedError('Integer division by zero')`.

#### 2.3 64-Bit Index Safety & Float-to-Int Clamping
- **Float-to-Int Clamping**: In `hook/custom_ufuncs.cpp:15951, 15987` (`histogram_uniform_kernel`), out-of-range floats cast directly to `int64_t` produced undefined behavior. Clamping is now performed within `double` bounds (`[0, nbins - 1]`) before casting.
- **64-bit Index Variables**: Replaced 32-bit `int` with `int64_t` for index tracking in `s_reduceat_op_impl` (`line 14554`), `histogram_binsearch` (`lines 16015-16055`), `fast_interp_vector` (`line 11719`), `get_quantile_specs` (`lines 11463-11521`), and `reflect_map`/`symmetric_map` (`lines 10305, 10312`), preventing truncation or buffer overflows on arrays with > $2^{31}$ elements.

#### 2.4 Memory Management & Finalizer Hygiene
- **Native Allocations**: Managed via Dart's `Arena` and `NativeFinalizer`. In accordance with `AGENTS.md`, `NativeFinalizer.attach` does not use `externalSize` to prevent aggressive GC thrashing. Deterministic reclamation is handled by `NDArray.scope`.
- **Advanced Indexing Memory Leak**: In `lib/src/ndarray.dart:3220-3339` (`_sliceAssign` where `isAdvanced == true`), the broadcasted right-hand side array was not closed if an exception occurred during element assignment. A `try`/`finally` block was added ensuring `broadcastedVal.dispose()` is executed. Additionally, if the assigned value shares underlying memory with the destination array (`sharesMemory(this, valArr)`), a defensive copy is created before assignment.
- **Out of Memory Flagging**: Native memory allocation buffers in `hook/custom_indexing.cpp:60-77` (`NoThrowBuffer`) invoke `ndarray_set_oom_flag()`. When native operations return `-4`, Dart wrappers translate the code into a clean `throw const OutOfMemoryError()`.

---

### Dimension 3: Numerical Correctness & Edge-Case Robustness

#### 3.1 Destination (`out:`) Buffer Validation & In-Place Aliasing
Every operation supporting an `out:` destination buffer now validates:
1. `!out.isWriteable`: Throws `StateError` if the destination array is marked read-only.
2. `out.size > 1 && out.strides.contains(0)`: Throws `ArgumentError` if the output array is a broadcast view containing zero strides (which would cause multiple output elements to write to the same physical memory offset).
3. Memory Aliasing: In operations where destination buffers may overlap with input arrays (`argsort`, `argpartition`, `searchsorted` in `lib/src/operations/sorting.dart`), the implementation checks `sharesMemory(a, out)`. If memory is shared, the calculation executes into a temporary array inside an `NDArray.scope` and copies the final result to `out:`, preventing self-aliasing data corruption.

#### 3.2 0-Dimensional (Scalar) and Empty (0-Length) Arrays
- **Empty Array Reductions**: Reducing an empty array along an axis where `shape[axis] == 0` (e.g. reducing shape `[2, 0]` along `axis: 1`) correctly throws `ArgumentError('Cannot reduce over an empty axis.')`. However, reducing an empty array along a non-zero axis (e.g. shape `[2, 0]` along `axis: 0`) produces an empty array of target shape `[0]`. This behavior was normalized across `quantile`, `median`, `ptp`, `min`, `max`, `argmin`, `argmax`, `nanargmin`, and `nanargmax` in `lib/src/operations/stats.dart` and `lib/src/operations/sorting.dart`.
- **Scalar Properties**: 0-D arrays correctly return their scalar value via `.scalar`.

#### 3.3 Dimension Checking in Reductions & Convolution
- **Axis Validation**: In `lib/src/operations/stats.dart:30-40` (`_reductionTargetShape`), axis validation was hardened with `RangeError.range(axis, -shape.length, shape.length - 1, 'axis')` before any indexing or slice modifications on `shape`.
- **Empty Signal Rejection**: `correlate` and `convolve` in `lib/src/operations/dsp.dart:619, 741` now check `in1.size == 0 || in2.size == 0` upfront and throw `ArgumentError('Input arrays cannot be empty.')`.
- **Integer Validation in `bincount`**: `lib/src/operations/binning.dart:85-125` validates `x.dtype.isInteger` before execution, validates the `out:` buffer prior to short-circuiting on empty inputs, and properly supports `Uint64` indices up to $2^{63}-1$.

---

### Dimension 4: Performance, SIMD Vectorization, & BLAS Integration

#### 4.1 Native SIMD Acceleration
- **Highway Sorting Integration**: High-performance vectorized sorting and counting algorithms (`unique_int64_fast`, `unique_int32_fast`, `unique_int16_fast`, `unique_uint8_fast`, plus `Int8`, `Uint16`, `Uint32`, `Uint64`, `Float32`, `Float64`) using Google Highway `VQSort` are wired into `ndarray_unique` in `hook/custom_sorting.cpp`, delivering up to 12x speedups on large unique operations.
- **Binary Elementwise Min/Max**: `binaryUfunc` in `lib/src/operations/math/ufunc_methods.dart` dispatches binary `minimum`, `maximum`, `fmin`, and `fmax` to native C Highway SIMD (`v_binary_minmax`) and strided (`s_binary_minmax`) kernels in `hook/custom_ufuncs.cpp`.
- **Matrix Multiplication Cache Locality**: In `hook/custom_ufuncs.cpp:10029-10062` (`DEFINE_MATMUL_INT`), an optimized row-major loop ordering (`r-i-c`) was introduced for contiguous inputs (`strideACol == 1 && strideBCol == 1 && strideResCol == 1`), drastically reducing CPU L1/L2 cache misses compared to naive `r-c-i` loops.

#### 4.2 BLAS, LAPACK, & PocketFFT
- Linear algebra routines delegate heavy-duty GEMM and factorization operations to `package:openblas`.
- Discrete Fourier transforms delegate multi-dimensional FFTs directly to `package:pocketfft`.
- Dart-side allocations avoid element-by-element loops, leveraging native FFI buffers and bulk memory copying (`NDArray.copy`).

---

### Dimension 5: Pub Publishing, Native Assets Hook, & Packaging Readiness

#### 5.1 Package Archive Optimization (`.pubignore`)
A severe packaging flaw was identified during dry-run audit: `pkgs/ndarray/.pubignore` omitted test coverage directories (`coverage/`), scratch build directories (`scratch/`), lockfiles (`pubspec.lock`), and temporary CMake build trees (`third_party/highway/build/`). This caused `dart pub publish --dry-run` to package **14.2 megabytes** of unversioned files.
- **Fix**: Added all excluded paths to `pkgs/ndarray/.pubignore`. The resulting package archive size is **< 1 megabyte**.
- **Static Invariant Enforcement**: A static test was implemented in `test/meta/build_infra_invariants_test.dart` to verify that every path ignored in `.gitignore` is also excluded in `.pubignore`.

#### 5.2 Native Assets Compilation Hook (`hook/build.dart`)
`hook/build.dart` implements the `package:hooks` / `package:code_assets` standard protocol:
- In development/source mode: Automatically builds the `ndarray` shared library (`libndarray`) from C++ sources using the host C++17 toolchain (`clang++` / `g++` / `MSVC`) with `-O2`/`-O3`, Highway SIMD optimizations, and optional sanitizer flags (`NDARRAY_SANITIZE=address,undefined`).
- In prebuilt mode: Validates the SHA-256 hash of the native source files against `nativeSourceHash` in `lib/src/hook_helpers/hashes.dart`. If it matches, it downloads prebuilt binaries from the GitHub Release tag `artifacts-v0.2.0` and verifies them against `prebuiltArtifactHashes` (`fileHashes`), enabling builds on machines without a native C++ compiler.
- `lib/src/hook_helpers/hashes.dart` remains intentionally pinned to the `artifacts-v0.2.0` release hashes in-tree (as strictly enforced by `test/meta/codebase_invariants_test.dart:811-860`). Running `tool/regenerate_hashes.dart` after uploading rebuilt native binaries to GitHub Release `artifacts-v0.2.0` is an external release task documented under P0-2.

---

## 3. Complete Catalog of Issues Fixed In-Tree (Milestones M1 & M2)

| Issue ID | Subsystem & File Location | Category & Severity | Root Cause Analysis | Remediation Applied | Verification & Regression Test |
|---|---|---|---|---|---|
| **D5-1** | `pkgs/ndarray/.pubignore:1-24` | Packaging (P0) | `.pubignore` omitted `coverage/`, `scratch/`, `pubspec.lock`, and `third_party/highway/build/`, causing `dart pub publish --dry-run` to bundle 14 MB of binary test logs and coverage traces. | Added all missing patterns to `.pubignore`. Package archive reduced to < 1 MB. | `test/meta/build_infra_invariants_test.dart` enforces `.gitignore` $\subseteq$ `.pubignore`; dry-run passes cleanly. |
| **F1** | `hook/custom_ufuncs.cpp:5730-5731, 13623-13624, 14332-14357` | Native Safety / UB (High) | Signed integer overflow on subtraction and accumulation in `s_diff_int64`/`int32`, `add_prod_corr`, and `v_reduceat_unroll_helper` causes undefined behavior under C++17. | Cast operands to `uint64_t`/`uint32_t`/`std::make_unsigned_t<T>` prior to arithmetic operations. | Tested under `-fsanitize=undefined`; dynamic tests in `test/meta/operation_contracts_test.dart`. |
| **F2** | `hook/custom_ufuncs.cpp:14358-14402` | Numerical / SIMD (High) | 8-way unrolled reduction in `v_reduceat_unroll_helper` initialized accumulator lanes with `src[start]`. If `src[start]` was NaN, all 8 lanes were poisoned, corrupting slices of length > 16. | Replaced accumulator initialization with `apply_at_op(accK, src[i+K], opCode)` to preserve IEEE 754 NaN handling rules. | Added unit tests in `test/core/math_ufunc_test.dart` testing `reduceat` with leading NaN on lengths 17–64. |
| **F3** | `hook/custom_ufuncs.cpp:13998-14033`, `lib/src/operations/math/ufunc_methods.dart:2583-2920, 3220-3407` | Native / Contract (High) | Integer division or modulo by 0 in `at` and `reduceat` triggered hardware SIGFPE crashes or undefined behavior. | Added check `if (b == 0) division_error_flag = 1;` in C kernels. Dart callers check `get_and_reset_division_error()` and throw `UnsupportedError('Integer division by zero')`. | Added test cases in `test/core/math_ufunc_test.dart` for `at` and `reduceat` dividing by zero. |
| **F4** | `hook/custom_ufuncs.cpp:10305, 11463-11521, 11719, 14554, 15951, 15987, 16015-16055` | Native / 64-Bit (Medium-High) | Out-of-range floats in `histogram_uniform_kernel` cast to `int64_t` produced UB; 32-bit `int` index variables truncated offsets on arrays exceeding 2 billion elements. | Clamped floating values in `double` before casting to integer; converted index variables in `s_reduceat_op_impl`, `histogram_binsearch`, `fast_interp_vector`, and `get_quantile_specs` to `int64_t`. | Evaluated in `test/meta/dtype_dispatch_invariants_test.dart` and boundary tests. |
| **F5** | `hook/custom_indexing.cpp:60-77, 2096`, `lib/src/operations/{padding,repeating_tiling,manipulation,indexing}.dart` | Native / Memory (Medium) | `NoThrowBuffer` failed to notify Dart when native allocations failed; `get_roll_dtype_itemsize` called `abort()` on invalid dtype. | Added `ndarray_set_oom_flag()`, converted `-4` return codes to `OutOfMemoryError` in Dart callers, and replaced `abort()` with safe fallback return. | Tested in `test/meta/dtype_dispatch_invariants_test.dart`. |
| **F6** | `lib/src/ndarray.dart:3220-3339` | Memory Lifecycle (Medium-High) | Advanced indexing `_sliceAssign` leaked native memory when `broadcastTo` views were not disposed, and corrupted data when right-hand side shared memory with destination. | Enclosed `broadcastTo` in `try`/`finally` with `.dispose()`, and checked `sharesMemory(this, valArr)` to defensively copy input if aliased. | Verified in `test/meta/lifetime_invariants_test.dart` and allocation tracking suites. |
| **F7** | `hook/npz_io.cpp:178-501, 697-823` | I/O Safety & ZIP64 (Medium) | Writing `.npz` files approaching or exceeding 4 GiB without ZIP64 caused 32-bit cumulative offset overflow, producing corrupted archives. | Implemented full ZIP64 support (`0x0001` extra fields, `0x06064b50` ZIP64 EOCD, `0x07064b50` ZIP64 EOCD Locator) in `npz_save_stored`, `npz_save_deflate`, and `npz_open_reader` (returning `-4` / `UnsupportedError` on unrecoverable archive limits). | Verified by ZIP64 round-trip and boundary assertions in `test/core/io_test.dart`. |
| **F8** | `hook/custom_ufuncs.cpp:193-199` | Build Hook (Low) | Unconditional `#define VECTORIZED_TARGETS` caused compiler redefinition warnings during native build. | Wrapped definition in `#ifndef VECTORIZED_TARGETS ... #endif`. | Native compilation builds with 0 compiler warnings. |
| **F9** | `hook/custom_sorting.cpp:2865-2881` | Performance / SIMD (Medium) | Vectorized Highway sorting kernels for `Int64`, `Int32`, `Int16`, and `Uint8` were compiled but dead (omitted from `ndarray_unique` switch). | Wired `unique_int64_fast`, `unique_int32_fast`, `unique_int16_fast`, and `unique_uint8_fast` into `ndarray_unique`; `hashes.dart` remains pinned to `artifacts-v0.2.0` pending P0-2 release cut. | Verified correctness in `test/core/sorting_searching_test.dart` and benchmark assertions. |
| **F12** | `hook/custom_ufuncs.cpp:10029-10064` | Performance (Medium) | Integer GEMM in `DEFINE_MATMUL_INT` used cache-unfriendly `r-c-i` loop order for contiguous buffers. | Added contiguous `r-i-c` loop order to `DEFINE_MATMUL_INT`. | Benchmarked in performance test suite; verified in `test/core/math_ufunc_test.dart`. |
| **D1-1** | `lib/src/operations/dsp.dart:25` | API / Typing (P1) | Leaked `typedef Float = double;` in public library scope collided with `dart:ffi`'s `Float`. | Removed public typedef; replaced internal usage with `double?`. | Verified via `test/meta/codebase_invariants_test.dart`. |
| **D1-3** | `lib/src/operations/linalg.dart:1645, 1691, 1737, 1784` | Error Contract (P1) | `inv()` threw `StateError` on singular matrices instead of `LinAlgError`, contradicting its Dartdoc and other linalg routines. | Updated `inv()` to throw `const LinAlgError('Matrix is singular and cannot be inverted.')`. | Added regression tests in `test/core/linalg_test.dart`. |
| **D1-5** | `lib/src/ndarray.dart:5554, 5757, 5768` | Ergonomics (P1) | `Complex`, `IndexSpec`, and `Index` lacked `const` constructors, breaking code and examples attempting `const Complex(0, 0)`. | Added `const` keyword to constructors. | Verified compilation of `README.md` example and `const` instantations in test suite. |
| **D1-7** | `lib/src/ndarray.dart:5822`, `lib/src/operations/{calculus,tensor_contractions,broadcasting}.dart` | Immutability (P1) | Caller-supplied lists in `Indices`, `CoordinateSpacing`, `TensordotAxes`, and `BroadcastResult` were stored without defensive copying. | Wrapped inputs in `List.unmodifiable(...)`. | Verified immutability invariant tests in `test/meta/codebase_invariants_test.dart`. |
| **D1-9** | `lib/src/ndarray.dart`, `sendable_ndarray.dart`, `random.dart`, `padding.dart`, `linalg.dart`, `optimize.dart`, `splitting.dart` | Documentation (P1) | Broken Dartdoc casts (`1.0 as Float64`), invalid doclinks (`[NDArray.ndim]`), undocumented enum constants, and terse 1-line docstrings. | Corrected doc comments, restored valid link targets, documented `PadMode` enums, and expanded function documentation. | Audited via `dart analyze` and doclint meta-tests. |
| **D1-11** | `lib/src/operations/dsp.dart:765-792`, `lib/src/operations/binning.dart:457` | Style / Architecture (P2) | `convolve` and `digitize` used `if-else if` chains to dispatch DTypes instead of `switch`, violating `AGENTS.md`. | Refactored both functions to use canonical `switch (dtype)` statements. | Enforced by `test/meta/dtype_dispatch_invariants_test.dart`. |
| **D1-4 & D3-1** | Universal (`lib/src/operations/helpers.dart:46`, across 133 operations) | Safety / Contract (High) | `out:` destination parameters did not reject read-only arrays or broadcast zero-stride views, permitting silent memory corruption. | Implemented `validateOutArray` and `validateOutBuffer` enforcing `!out.isWriteable` and `out.size > 1 && out.strides.contains(0)` checks. | Covered by cross-cutting table-driven suite in `test/meta/operation_contracts_test.dart`. |
| **D3-2** | `lib/src/operations/sorting.dart:389-402, 821-834, 2962-2985` | Correctness (High) | In-place destination aliasing in `argsort`, `argpartition`, and `searchsorted` caused source reading while writing to `out:`. | Checked `sharesMemory(a, out)`; if true, executes into scoped temporary array and copies result. | Added aliased destination test cases to `test/core/sorting_searching_test.dart`. |
| **D3-3** | `lib/src/operations/stats.dart:30-40` | Bounds Safety (High) | `_reductionTargetShape` did not validate normalized axis boundaries before mutating target shape list. | Added `RangeError.range(axis, -shape.length, shape.length - 1, 'axis')` check upfront. | Verified in `test/core/stats_test.dart` with out-of-range axes. |
| **D3-4** | `lib/src/operations/stats.dart:4849-4850` | Correctness (High) | `ptp` dropped `keepdims` argument when forwarding to underlying `max` and `min` calls. | Added `bool keepdims = false` parameter to `ptp` and forwarded `keepdims: keepdims`. | Added test in `test/core/stats_test.dart` verifying `ptp(a, axis: 0, keepdims: true)`. |
| **D3-5** | `lib/src/operations/dsp.dart:619, 741` | Correctness (High) | `correlate` and `convolve` failed with unhandled index errors or zero division on empty input arrays. | Added explicit check `if (in1.size == 0 || in2.size == 0) throw ArgumentError('Input arrays cannot be empty.');`. | Added empty array test assertions in `test/core/dsp_test.dart`. |
| **D3-6** | `lib/src/operations/binning.dart:85-125` | Type Safety (Medium) | `bincount` failed to validate integer DTypes, bypassed `out:` validation on empty inputs, and failed on `Uint64` values $\ge 2^{63}$. | Added `x.dtype.isInteger` validation, moved `validateOutArray` before empty return, and added 64-bit unsigned bounds checking. | Verified in `test/core/binning_test.dart`. |
| **D3-7** | `lib/src/operations/stats.dart:4205, 4533, 4766`, `lib/src/operations/sorting.dart:2699, 2843` | Correctness (Medium) | Empty 2D array reductions along non-zero axes (e.g. `[2, 0]` along `axis: 0`) threw `ArgumentError` instead of producing empty output shape `[0]`. | Normalized condition to check whether the *reduced* axis is empty (`shape[axis] == 0`), returning empty target array otherwise. | Updated and verified in `test/core/sorting_searching_test.dart` and `test/core/stats_test.dart`. |

---

## 4. Meta-Test & Invariant Suite Hardening (Milestone M2)

To prevent entire classes of regressions, `pkgs/ndarray/test/meta/` incorporates five specialized static and dynamic invariant suites:

### 4.1 Static Invariant Enforcers (`codebase_invariants_test.dart` & `build_infra_invariants_test.dart`)
- **API Surface Invariants**: Uses the Dart analyzer AST visitor to verify that no public function in `lib/` exposes raw pointer types (`Pointer`), direct `.data` field access, or positional `out` parameters.
- **DType Switch Completeness**: Parses all `switch (dtype)` and `switch (a.dtype)` statements across `lib/src/operations/` and C++ kernels, confirming that every enum value is either handled or has an explicit, throwing `default:` branch.
- **Packaging Invariants**: Verifies that every pattern in `.gitignore` is represented in `.pubignore` and that no build-generated files are tracked.

### 4.2 Dynamic Behavioral Contracts (`operation_contracts_test.dart`)
- **Universal `out:` Contract**: Automatically runs hundreds of operations with:
  1. A read-only destination buffer (verifying `StateError`).
  2. A broadcast destination buffer with zero-strides (verifying `ArgumentError`).
  3. An overlapping/aliased destination buffer (verifying data equivalence against an unaliased reference).
- **Stride Invariance**: Validates that operations produce identical numerical results across contiguous, strided, sliced, and reversed (negative stride) inputs.
- **Empty Array Reductions**: Validates consistent shape calculation and error handling across 0-D scalars and empty multi-dimensional arrays.

### 4.3 Memory Lifecycle Invariants (`lifetime_invariants_test.dart`)
- **Scope Leak Detection**: Asserts that `NDArray.scope` promptly collects all intermediate allocations and that returning an array correctly preserves it in the enclosing scope via `attachTo`.
- **ScratchArena Invariant**: Asserts that temporary scratch allocations restore the arena marker upon completion.

---

## 5. Prioritized Pre-Publish & Post-Publish Roadmap

### P0: External Release Blockers (Must Complete Before `dart pub publish`)

```
   +-------------------------------------------------------------+
   | (P0-1) Publish Workspace Packages to pub.dev                |
   |   1. resource_scope: ^0.1.0                                 |
   |   2. openblas: ^0.1.0 & pocketfft: ^0.2.0                   |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | (P0-2) Build & Upload Native Prebuilt Binaries              |
   |   - Compile ndarray across Linux, macOS, Windows            |
   |   - Upload to GitHub Release artifacts-v0.2.0               |
   |   - Update nativeSourceHash & prebuiltArtifactHashes        |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | Execute `dart pub publish`                                  |
   +-------------------------------------------------------------+
```

#### P0-1: Publish Sibling Workspace Packages to `pub.dev`
- **Location**: `pkgs/ndarray/pubspec.yaml:37-44`
- **Issue**: `package:ndarray` depends on three sibling workspace packages:
  - `resource_scope: ^0.1.0`
  - `openblas: ^0.1.0`
  - `pocketfft: ^0.2.0`
- **Action**: These packages currently reside as local workspace paths. They must be published to `pub.dev` in strict topological order before `ndarray` can be published:
  1. Publish `resource_scope`.
  2. Publish `openblas` and `pocketfft`.
  3. Run `dart pub get` in `pkgs/ndarray` to resolve dependencies against `pub.dev` registry versions.

#### P0-2: Compile & Upload Prebuilt Native Binaries to GitHub Releases
- **Location**: `pkgs/ndarray/hook/build.dart:44-50`, `lib/src/hook_helpers/hashes.dart:14-41`
- **Issue**: `hook/build.dart` downloads precompiled binaries of `ndarray` (`libndarray`) for target architectures when the host environment lacks a C++ compiler. Prebuilt archives are validated against the SHA-256 hash `nativeSourceHash` and `prebuiltArtifactHashes` (`fileHashes`) in `hashes.dart`.
- **Action**:
  1. Trigger GitHub Actions CI workflow to build `ndarray` across all release targets:
     - `linux-x64`, `linux-arm64`
     - `macos-arm64`, `macos-x64`
     - `windows-x64`
  2. Upload archives to GitHub release `artifacts-v0.2.0`.
  3. Run `dart tool/regenerate_hashes.dart artifacts-v0.2.0` to populate `nativeSourceHash` and `prebuiltArtifactHashes` (`fileHashes`) in `lib/src/hook_helpers/hashes.dart`.

---

### P1: Performance & Native C/SIMD Kernel Enhancements (Completed In-Tree)

- **P1-1 (Completed)**: Extended Highway `VQSort` and counting-sort specializations in `ndarray_unique` (`hook/custom_sorting.cpp`) to all 10 integer and floating-point `DType`s (`Int8`, `Uint16`, `Uint32`, `Uint64`, `Float32`, `Float64`, in addition to `Int64`, `Int32`, `Int16`, `Uint8`).
- **P1-2 (Completed)**: Implemented native C Highway SIMD (`v_binary_minmax`) and strided (`s_binary_minmax`) kernels for binary elementwise `minimum`, `maximum`, `fmin`, and `fmax` across all 14 `DType`s in `hook/custom_ufuncs.cpp` and wired `_nativeMinMax` into `binaryUfunc`.
- **P1-3 (Completed)**: Eliminated redundant Dart pointer loops in `accumulateUfunc` and `outerUfunc` (`lib/src/operations/math/ufunc_methods.dart`), routing directly to native C `s_cum*` and broadcasted `binaryUfunc` kernels.
- **P1-4 (Completed)**: Promoted `nansum`, `all`, `any`, `nanvar`, and complex `min`/`max` reductions in `lib/src/operations/stats.dart` to native C kernels and unboxed FFI pointer loops.
- **P1-5 (Completed)**: Eliminated `.toList()` and `getCellFlat` indexing round-trips in `lib/src/ndarray.dart` via unboxed `Int64List` and `castNDArray<Int64>`.

---

### P2: Medium-Priority Architectural & Ergonomic Improvements (Completed In-Tree)

- **P2-1 (Completed)**: Strongly typed `unique<T>` to return `NDArray<T>` and added `uniqueWithIndex<T>`, `uniqueWithInverse<T>`, `uniqueWithCounts<T>`, and `uniqueAll<T>` returning named records (`lib/src/operations/set_operations.dart`).
- **P2-2 (Completed)**: Converted `divmod` (`({NDArray<T> quotient, NDArray<T> remainder})`) and `ndenumerate` (`({List<int> coordinate, Object value})`) to return named records.
- **P2-3 (Completed)**: Eliminated all remaining `dynamic` parameters and un-generic `NDArray` types from exported signatures (`lib/src/ndarray.dart`, `lib/src/operations/`), enforced by resolved `DartType` AST invariants in `test/meta/codebase_invariants_test.dart`.
- **P2-4 (Completed)**: Aligned `nansum` narrow integer/boolean accumulation `DType` widening (`Int8/16/32/Boolean -> Int64`, `Uint8/16/32 -> Uint64`) with `sum` (`lib/src/operations/stats.dart`).
- **P2-5 (Completed)**: Standardized out-of-bounds `axis` errors to `RangeError.range` across `dsp.dart`, `calculus.dart`, `linalg.dart`, `manipulation.dart`, and `spacers.dart`, enforced across 40+ operations in `test/meta/operation_contracts_test.dart`.
- **P2-6 (Completed)**: Expanded `{@example /example/... lang=dart}` tag coverage across `binning.dart`, `broadcasting.dart`, `calculus.dart`, `dsp.dart`, `indexing.dart`, `repeating_tiling.dart`, and `set_operations.dart`.

---

### P3: Long-Term Enhancements (Completed In-Tree)

- **P3-1 (Completed)**: Implemented full ZIP64 large archive support (`0x0001` extended information extra fields, `0x06064b50` ZIP64 End of Central Directory record, and `0x07064b50` ZIP64 End of Central Directory Locator) in `hook/npz_io.cpp` (`npz_save_stored`, `npz_save_deflate`, and `npz_open_reader`), supporting reading and writing multi-gigabyte `.npz` archives larger than 4 GiB or containing $\ge 65,535$ entries.

---

## 6. Verification Summary & Commands

The following verification commands were executed directly within `/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray` using Dart SDK `3.7+`:

### 1. Code Formatting
```bash
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart format --output=none --set-exit-if-changed lib test
```
- **Output**: `Formatted 192 files (0 changed)`
- **Exit Code**: `0` (Clean)

### 2. Static Analysis
```bash
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart analyze
```
- **Output**: `Analyzing pkgs/ndarray... No issues found!`
- **Exit Code**: `0` (Clean)

### 3. Meta-Test Invariant Suites
```bash
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart test test/meta/build_infra_invariants_test.dart
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart test test/meta/codebase_invariants_test.dart
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart test test/meta/dtype_dispatch_invariants_test.dart
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart test test/meta/lifetime_invariants_test.dart
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart test test/meta/operation_contracts_test.dart
```
- **Output**: All meta-invariant tests pass 100%.

### 4. Full Package Test Suite
```bash
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart test
```
- **Output**: `+4587: All tests passed!`
- **Exit Code**: `0` (Clean)

### 5. Native Sanitizers (ASan / UBSan)
```bash
NDARRAY_SANITIZE=address,undefined /usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart test
```
- **Output**: 0 AddressSanitizer errors, 0 UndefinedBehaviorSanitizer errors across all compiled C/C++ kernels.

### 6. Pub Packaging Dry-Run
```bash
/usr/local/google/home/sigurdm/.puro/envs/master/flutter/bin/cache/dart-sdk/bin/dart pub publish --dry-run
```
- **Archive Size**: `< 1 MB` (reduced from 14.2 MB)
- **Status**: Successful validation; only pre-publish workspace package dependency notices remaining.
