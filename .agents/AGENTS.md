Remember when adding new operations, fixing old ones:
* always allow for an out: argument whereever it makes sense
* always allow for complex float and integer inputs where it makes sense.
* Always use strong typing and generics for all NDArray arguments.
* always use ffi with C implementation. Use intrinsics to provide optimized flat contiguous version for speed. Try to be smart about this.
* always allow for strided version of the operation
* always use a switch to dispatch to the correct implementation based on DType rather than a if-else if chain.
* Documentation should be rich and detailed, in the same style as numpy.
* openblas and lapack bindings belong in the openblas package.
* Always use ScratchArena for temporary allocations.
* Avoid using setRange and toList handling ndarrays in dart space. Rather make views and use NDArray.copy.
* When returning multiple values use records with named fields instead of HashMaps.
* It is usally ok for the result of an operation to have same dtype as the input. We like the conversions to be explicit.
* Avoid accessing or indexing `NDArray.data` directly in Dart code (as it is `@internal`).
* Use the `.scalar` getter to access the value of 0-dimensional arrays.
* Always prefer NDArray<Float64> or NDArray<Float32> over NDArray<double> for argument and return values.
* Whenever applicable, use NDArray.scope instead of manually calling dispose. Remember that results must be attached to the parent scope before returning.
* Do NOT pass `externalSize` to `NativeFinalizer.attach` for `NDArray` buffers. `externalSize` is too blunt a tool and can cause severe GC thrashing on large allocations; always recommend and use `NDArray.scope` (or manual `dispose()`) for prompt reclamation.
* Always use enums for options/modes instead of magic strings where NumPy or other APIs accept string options.
* When adding tests or investigating bugs, always consider ways of eliminating entire classes of bugs across the codebase instead of only validating a point regression in a single location:
  - Add new operations and cross-cutting behavioral invariants to `pkgs/ndarray/test/meta/operation_contracts_test.dart` and table-driven test suites.
  - Encode structural, FFI, and API/Dartdoc rules as static AST/source checks in `pkgs/ndarray/test/meta/codebase_invariants_test.dart`.
  - Use instrumentation and dynamic analysis tools—such as native C/C++ sanitizers (`sanitize: undefined` or `sanitize: address,undefined` under `hooks.user_defines.ndarray` in `pubspec.yaml`, or `NDARRAY_SANITIZE=...` when invoking build hooks directly), `NDArray` allocation tracking (`trackAllocations`), and `ScratchArena` marker assertions—to catch memory, undefined-behavior, and resource-lifecycle bugs systematically.


When running dart commands use the sdk specified in .vscode/settings.json.