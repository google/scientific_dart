# resource_scope

Zone-based automatic scoped resource management for deterministic FFI and native memory disposal in Dart.

## Features

- **Automatic Resource Disposal**: Execute code blocks inside `ResourceScope.scope(() { ... })`, and any object implementing `ScopedResource` created inside that zone will be deterministically disposed (`.dispose()`) when the scope exits.
- **Scope Escaping / Return Promotion**: Use `ResourceScope.returning(() { ... })` or `.detachFromScope()` / `.detachToParentScope()` to safely return resources to the caller or parent scope while cleaning up all intermediate allocations.
- **Hybrid Collection Scaling**: Tracks small scopes (up to 100 elements) in a fast contiguous `List` for CPU cache locality and zero hashing overhead, automatically promoting to a `HashSet` ($O(1)$ scaling) for larger collections.
- **Zero Dependencies**: Lightweight and standalone so any FFI wrapper, database client, audio/video stream library, or math package can implement `ScopedResource`.

## Usage

```dart
import 'package:resource_scope/resource_scope.dart';

final class NativeBuffer implements ScopedResource {
  bool _disposed = false;

  NativeBuffer() {
    // 1. Opt into automatic zone tracking
    ResourceScope.track(this);
  }

  @override
  bool get isDisposed => _disposed;

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    // Free native memory...
    ResourceScope.untrack(this);
  }

  @override
  ScopedResource detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  ScopedResource detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }
}

void main() {
  // Scoped execution automatically frees buffers at block end
  ResourceScope.scope(() {
    final b1 = NativeBuffer();
    final b2 = NativeBuffer();
  }); // b1 and b2 are deterministically disposed here

  // Returning a resource out of a scope
  final survived = ResourceScope.returning(() {
    final temp = NativeBuffer();  // Disposed
    final out = NativeBuffer();   // Promoted out of scope
    return out;
  });
}
```
