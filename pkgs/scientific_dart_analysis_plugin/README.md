# scientific_dart_analysis_plugin (Experimental)

> [!WARNING]
> **Experimental**: This analyzer plugin is experimental and under active development. Rule names, diagnostics, and quick fixes may change across versions.

An analyzer plugin (`package:analysis_server_plugin`) providing static analysis lints and quick fixes for `scientific_dart` (`package:ndarray`, `package:resource_scope`, `package:symbolic_dart`) consumers.

## Features

- **Memory & Scope Safety**: Detects unescaped `NDArray.scope` returns, view lifecycle misuse, and missing `detachToParentScope()` calls.
- **API & Correctness Lints**: Flags `==` equality misuse on `NDArray` (suggesting `equal()` or `allClose()`), broadcast views passed as `out:` targets, and unsigned 64-bit comparison pitfalls (`uint64Compare`).
- **Performance Lints**: Highlights avoidable `.toList()` materializations and repeated `.lambdify()` compilation inside loops.

## License

This package is licensed under the **[Apache License, Version 2.0](LICENSE)**.
