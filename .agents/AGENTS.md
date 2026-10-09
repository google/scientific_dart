# Scientific Dart Workspace Guidelines

## 1. API Design & Static Typing (`DTypeTag` / `DTypeSpec`)
* **Strong typing with `DTypeTag`, `SelfOf`, & `DTypeSpec`:** Avoid widening argument or return types to bare `NDArray` / `NDArray<DTypeTag>`. Use:
  - `<T extends DTypeTag>` for single-input dtype-preserving operations,
  - `<T extends SelfOf<DTypeTag>>` (or `SelfOf<...Of<R>>`, `SelfOf<RealOf<BitwiseDType>>`, `SelfOf<RealOf<IntegerDType>>`) whenever one type parameter binds two or more input array positions (two `NDArray<T>` parameters, or a `List<NDArray<T>>`), so that distinct concrete dtype tags (`LUB = DTypeTag`, `BitwiseDType`, `IntegerDType`, or a shared projection interface, none of which implement `SelfOf`) are rejected symmetrically at compile time while keeping a single type parameter `<T>`. Do not use `SelfOf` on a parameter that binds a single input position (it only leaks `SelfOf<DTypeTag>` as an inferred type), and do not use it on operations that deliberately promote mixed dtypes at run time (`where`, `union1d`, `clipArray`, ...). `SelfOf<X>` does not imply `X`: a `SelfOf`-bounded `T` cannot be passed to the unary projection-bounded operations,
  - Single-slot projection interfaces (`RealOf<R>`, `ElementOf<E>`, `RealFloatOf<R>`, `ComplexOf<R>`, `InexactOf<R>`, `AccumulatorOf<R>`, `DoublePrecisionOf<R>`, `DivideOf<R>`) when the output dtype is a deterministic function of the input dtype (e.g., real part `RealOf`, element access `ElementOf`, float computation `RealFloatOf`, complex promotion `ComplexOf`, math/inexact promotion `InexactOf`, sum/product accumulation `AccumulatorOf`, double promotion `DoublePrecisionOf`, true division `DivideOf`),
  - Non-generic capability markers (`BitwiseDType`, `IntegerDType`) which pin the projections that are constant across the integer/boolean family (`RealFloatOf<Float64>`, `ComplexOf<Complex128>`, `InexactOf<Float64>`, `DoublePrecisionOf<Float64>`, `DivideOf<Float64>`, and `ElementOf<int>` for `IntegerDType`) and preserve them across generic code and least-upper-bound (LUB) operations; they cannot pin `RealOf`/`AccumulatorOf` (which differ per tag), so unary bitwise/shift operations are bounded by `RealOf<BitwiseDType>` / `RealOf<IntegerDType>` instead and run-time-typed arrays go through `asBitwiseDType` / `asIntegerDType`, whose `DTypeSpec` rows `AnyBitwiseSpec` / `AnyIntegerSpec` satisfy both the capability and the `SelfOf` bounds,
  - The 8-slot projection table `DTypeSpec<RealTag, Element, RealFloatTag, ComplexTag, InexactTag, AccumulatorTag, DoublePrecisionTag, DivideTag>` (which implements `SelfOf` and all eight single-slot interfaces) only for functions requiring two distinct projections of one input (`slogdet`, `svd`, `eigh`), or
  - a concrete tag (`NDArray<Float64>`, `NDArray<Int64>`, `NDArray<Boolean>`, etc.) when the output dtype is fixed.
* **Explicit dtype conversions:** Keep dtype conversions explicit. Unless an operation's contract specifies promotion (such as transcendental math promoting integers/half-floats via `InexactTag`, or `Boolean` reductions accumulating into `Int64` via `AccumulatorTag`), preserve the input dtype.
* **64-bit index & count outputs:** Operations that return indices or counts (`argmax`, `argmin`, `argsort`, `argpartition`, `searchsorted`, `nonzero`, `argwhere`, `flatnonzero`, `count_nonzero`, `digitize`, `ravel_multi_index`, `unravel_index`, `unique` indices/counts) must always return `NDArray<Int64>`.
* **Full 15-dtype coverage:** Support all applicable `DType`s across floating-point (`float64`, `float32`, `float16`, `bfloat16`), complex (`complex128`, `complex64`), signed/unsigned integers (`int64`–`int8`, `uint64`–`uint8`), and `boolean`. Remember that Dart `int` is signed 64-bit, so `uint64` comparisons require `uint64Compare`.
* **Exhaustive `switch` dispatch:** Always use an exhaustive `switch (dtype)` to dispatch on `DType` rather than `if` / `else if` chains.
* **Named `{out}` parameter:** Allow a named `{NDArray<...>? out}` parameter wherever an operation produces an array. Validate `out.isWriteable`, `out.shape`, and `out.dtype`, and handle both non-contiguous `out` views and memory aliasing (`sharesMemory(input, out)`).
* **0-D scalar access:** Use the `.scalar` getter to read the value of 0-dimensional arrays.
* **Multi-value returns:** Return records with named fields (e.g. `({NDArray<T> values, NDArray<Int64> indices})`) instead of `Map`s or positional records.
* **Enums over strings:** Always use typed enums for options and modes instead of magic strings where NumPy accepts string options.

## 2. Memory Management, Views & `ScratchArena`
* **`NDArray.scope` first:** Use `NDArray.scope` instead of manual `dispose()` wherever applicable. Any array or view returned from a scope must be escaped via `.detachToParentScope()` (and note that detaching a view detaches its root owning buffer).
* **No `externalSize` on `NativeFinalizer`:** Do NOT pass `externalSize` to `NativeFinalizer.attach` for `NDArray` buffers. `externalSize` causes severe GC thrashing on large allocations; rely on `NDArray.scope` or manual `dispose()` for prompt reclamation.
* **`ScratchArena` discipline:** Use `ScratchArena` for temporary native allocations.
  - Every `final marker = ScratchArena.marker;` must be immediately paired with `try { ... } finally { ScratchArena.reset(marker); }`.
  - Ensure the type argument in `ScratchArena.allocate<T>(count * ffi.sizeOf<T>())` matches `ffi.sizeOf<T>()`, and never exceed the segment count passed to `ScratchArena.getStridedBuffer(ndim, segments)`.
* **Avoid Dart-space list copies & `NDArray.data` in operations:**
  - Never call `.toList()`, `.setRange()`, `.data`, or `.dataRaw` inside `lib/src/operations/`. `.data` exposes the raw backing buffer without applying `offsetElements`, `strides`, or `shape`.
  - Instead, construct views, use `NDArray.copy()`, pass FFI pointers to native kernels, or iterate with `NDIter`.
* **Pointer & offset invariants:**
  - `array.pointer` already incorporates `offsetElements`—never add `offsetElements` a second time when passing `array.pointer` to FFI.
  - `NDIter.getIndex` returns a raw buffer offset; always pair it with `getCellRaw` / `setCellRaw`, never `getCellFlat` / `setCellFlat`.

## 3. Native C/C++ Kernels & FFI (`hook/`)
* **Contiguous SIMD fast-path + strided fallback:** Implement operations in C/C++ via FFI (`@ffi.Native`). Provide an optimized contiguous fast-path (using SIMD intrinsics where beneficial) alongside a general strided implementation supporting arbitrary positive, negative, and zero strides.
* **Package boundaries:** OpenBLAS and LAPACK bindings belong in `pkgs/openblas`; PocketFFT bindings belong in `pkgs/pocketfft`. Never import `package:<other>/src/...` across workspace packages.
* **64-bit & LLP64 safety in C/C++:**
  - Always use `int64_t` in C/C++ headers/sources and `ffi.Int64` / `ffi.Pointer<ffi.Int64>` in Dart FFI bindings for shapes, strides, element counts, and indices (never 32-bit `int*`).
  - Never use bare `long` (which is 32-bit on Windows LLP64) or 32-bit `fseek` / `ftell`.
* **No `Struct`-by-value in `@ffi.Native` (`dart2wasm` FFI compatibility):** Never pass or return `ffi.Struct` subclasses (such as `cpx_t` or `cpx_f_t`) by value in `@ffi.Native` declarations or C/C++ entrypoints. `dart2wasm`'s `WasmFfiNativeTransformer` only supports primitive numeric and pointer types across the Wasm FFI boundary and crashes at compile time (`Null check operator used on a null value` in `getType`) on by-value structs. Always pass structs via `ffi.Pointer<...>` (e.g., return `void` and write complex scalar results into a caller-allocated `ffi.Pointer<cpx_t> out` from `ScratchArena`).
* **`-fno-exceptions` & OOM safety:**
  - Native hooks compile with `-fno-exceptions`. Never use `throw`, `std::vector`, `std::map`, `std::set`, `std::call_once`, `<iostream>`, or `printf`.
  - Use `new (std::nothrow)` or `NoThrowBuffer` for heap allocations, declare all exported symbols in their corresponding `.h` header with include guards, and use portable overloaded `std::` math functions (e.g. `std::sin`, not `std::sinf`).
  - Check the return code of every `int`-returning `native_*` `@ffi.Native` call in Dart (e.g., `-4` for OOM), and check `get_and_reset_division_error()` after native integer floor-division or modulo kernels.
* **Prebuilt artifact hashes (`hashes.dart`):** Never manually edit `nativeSourceHash` in `lib/src/hook_helpers/hashes.dart` when changing `hook/` sources (doing so disables the build-from-source fallback and serves stale binaries). Update hashes only at release time via `dart tool/regenerate_hashes.dart <release-tag>`.

## 4. Documentation & Examples
* **NumPy-grade Dartdoc:** Every public class, constructor, method, getter, top-level function, typedef, and enum value exported by `lib/ndarray.dart` must have rich `///` documentation describing parameters, preconditions, exceptions, complexity, and references.
* **External `@example` snippets:** In `lib/`, never use inline ```` ```dart ```` blocks in Dartdoc. Place runnable examples in `example/` and reference them on their own line with `/// {@example /example/<file>.dart lang=dart}`.

## 5. Testing, Static Invariants & Tooling
* **Eliminate classes of bugs systematically:** When fixing a bug or adding an operation, go beyond a point regression test:
  - Register new operations and cross-cutting behavioral invariants (0-D, empty, negative/broadcast strides, `out:` aliasing, non-contiguous `out:`, `where:` masks, dtype preservation) in `pkgs/ndarray/test/meta/operation_contracts_test.dart`.
  - Encode structural, FFI, C++, and API/Dartdoc rules as static AST/source checks in `pkgs/ndarray/test/meta/codebase_invariants_test.dart`.
  - Encode consumer-facing lint rules and quick fixes (e.g. scope escapes, view lifecycle misuse, `==` vs `.equals()`, `uint64` signed comparisons, `out:` broadcast views) in `pkgs/scientific_dart_analysis_plugin`.
  - Use dynamic instrumentation—native sanitizers (`sanitize: undefined` or `sanitize: address,undefined` under `hooks.user_defines.ndarray` in `pubspec.yaml`, or `NDARRAY_SANITIZE=...`), `NDArray` allocation tracking (`trackAllocations`), and `ScratchArena` marker assertions—to catch memory and UB bugs.
* **Dart SDK:** When running `dart` commands, use the SDK specified in `.vscode/settings.json`.