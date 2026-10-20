// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'dart:async' show Zone, runZoned;
import 'dart:collection' show HashSet;

final _scopeKey = Object();

/// An interface for resources (such as native FFI memory wrappers, file handles,
/// or external graphics/numerical buffers) that can be explicitly disposed
/// and automatically tracked by [ResourceScope].
abstract interface class ScopedResource {
  /// Whether this resource has already been disposed.
  bool get isDisposed;

  /// Explicitly releases native memory or resources immediately.
  void dispose();

  /// Removes this resource from its current automatic [ResourceScope] so it
  /// survives beyond the scope's lifetime.
  ScopedResource detachFromScope();

  /// Detaches this resource from the current inner [ResourceScope] and promotes
  /// it to the parent outer scope (if one exists).
  ScopedResource detachToParentScope();
}

/// Automatic zone-based scoped resource manager for [ScopedResource] instances.
///
/// Use [ResourceScope.scope] or [ResourceScope.using] to execute code blocks
/// where any created [ScopedResource] is automatically disposed when the scope
/// finishes.
///
/// Inside [ResourceScope.using], the active [ResourceScope] handle provides
/// [defer] to register LIFO cleanup callbacks and [escape] to promote results
/// to the parent scope (or to manual ownership if there is no outer scope).
///
/// Use [ResourceScope.returning] to automatically promote a single returned
/// [ScopedResource] to the outer parent scope while disposing all intermediate
/// allocations.
final class ResourceScope {
  final ResourceScope? _parentScope;
  final List<ScopedResource> _list = [];
  Set<ScopedResource>? _set;
  List<void Function()>? _defers;
  bool _isDisposing = false;
  bool _isClosed = false;

  ResourceScope._(this._parentScope);

  /// Whether allocation leak tracking is enabled via `--dart-define=TRACK_RESOURCE_ALLOCATIONS=true`
  /// or `--dart-define=TRACK_NDARRAY_ALLOCATIONS=true`.
  static const bool trackAllocations =
      bool.fromEnvironment('TRACK_RESOURCE_ALLOCATIONS') ||
      bool.fromEnvironment('TRACK_NDARRAY_ALLOCATIONS');

  static final Set<ScopedResource> _trackedAllocations = HashSet(
    equals: identical,
    hashCode: identityHashCode,
  );

  /// Returns a list of currently active root (undisposed) [ScopedResource]s
  /// when [trackAllocations] is enabled.
  static List<ScopedResource> get trackedAllocations =>
      _trackedAllocations.toList();

  /// Checks that all tracked [ScopedResource]s have been disposed.
  /// Throws a [StateError] if any undisposed resources remain.
  ///
  /// Returns `true` if no leaks are detected, allowing calls inside `assert`.
  static bool checkNoLeaks() {
    if (_trackedAllocations.isNotEmpty) {
      final leaks = _trackedAllocations.toList();
      throw StateError(
        'Detected ${leaks.length} undisposed ScopedResources:\n'
        "${leaks.map((r) => '  $r').join('\n')}",
      );
    }
    return true;
  }

  /// Clears the list of tracked allocations.
  static void clearTrackedAllocations() {
    _trackedAllocations.clear();
  }

  /// Registers [resource] with the active [ResourceScope] in [Zone.current], if any.
  ///
  /// Constructors of classes implementing [ScopedResource] should call
  /// `ResourceScope.track(this)` during initialization.
  static void track(ScopedResource resource) {
    final scope = Zone.current[_scopeKey] as ResourceScope?;
    scope?._track(resource);
    if (trackAllocations) {
      _trackedAllocations.add(resource);
    }
  }

  /// Removes [resource] from the active [ResourceScope] in [Zone.current], if any.
  static void untrack(ScopedResource resource) {
    final scope = Zone.current[_scopeKey] as ResourceScope?;
    scope?._untrack(resource);
    if (trackAllocations) {
      _trackedAllocations.remove(resource);
    }
  }

  /// Executes [callback] within an automatic resource management scope, passing
  /// the active [ResourceScope] handle to [callback].
  ///
  /// Any [ScopedResource] registered via [track] during the execution of
  /// [callback] is automatically disposed when the scope exits (or throws),
  /// unless promoted via [escape] or [ScopedResource.detachToParentScope].
  /// Callbacks registered with [defer] execute in last-in, first-out (LIFO)
  /// order before tracked resources are disposed.
  ///
  /// Supports both synchronous callbacks and asynchronous [Future]-returning
  /// callbacks.
  static R using<R>(R Function(ResourceScope scope) callback) {
    final parentScope = Zone.current[_scopeKey] as ResourceScope?;
    final scope = ResourceScope._(parentScope);
    return runZoned(() {
      R result;
      try {
        result = callback(scope);
      } catch (e) {
        try {
          scope._dispose();
        } catch (_) {
          // Preserve the primary exception from callback; all deferred
          // callbacks and resource disposals still ran inside _dispose().
        }
        rethrow;
      }

      if (result is Future) {
        return result.whenComplete(scope._dispose) as R;
      }
      scope._dispose();
      return result;
    }, zoneValues: {_scopeKey: scope});
  }

  /// Executes [callback] within an automatic resource management scope.
  ///
  /// Any [ScopedResource] registered via [track] during the execution of
  /// [callback] will be automatically disposed of when the callback finishes (or throws).
  ///
  /// Supports both synchronous callbacks and asynchronous [Future]-returning callbacks.
  static R scope<R>(R Function() callback) => using((_) => callback());

  /// Executes [callback] within an automatic resource management scope, automatically
  /// detaching and returning the resulting [ScopedResource] to the parent scope (if any).
  ///
  /// Any intermediate [ScopedResource]s created inside [callback] will be
  /// automatically disposed, while the returned [ScopedResource] survives the
  /// inner scope and is promoted to the caller's outer scope.
  static T returning<T extends ScopedResource>(T Function() callback) {
    return scope(() {
      final res = callback();
      return res.detachToParentScope() as T;
    });
  }

  /// Executes [callback] within an unmanaged context, preventing any created
  /// [ScopedResource]s from being registered in or disposed of by active scopes.
  static R unmanaged<R>(R Function() callback) {
    return runZoned(callback, zoneValues: {_scopeKey: null});
  }

  /// Promotes [resource] from the current active scope to its parent outer scope,
  /// or throws a [StateError] if there is no active scope.
  static void promoteToParent(ScopedResource resource) {
    final scope = Zone.current[_scopeKey] as ResourceScope?;
    if (scope == null) {
      throw StateError(
        'detachToParentScope() is only valid inside an active NDArray scope.',
      );
    }
    if (scope._isClosed || scope._isDisposing) {
      throw StateError('Cannot promote a ScopedResource from a closed scope.');
    }
    final wasTracked = scope._untrack(resource);
    if (wasTracked) {
      scope._parentScope?._track(resource);
    }
  }

  /// Registers [callback] to be executed when this scope exits.
  ///
  /// Deferred callbacks run in last-in, first-out (LIFO) order before the
  /// [ScopedResource] instances tracked by this scope are disposed. If a
  /// deferred callback throws, all remaining deferred callbacks and all
  /// tracked [ScopedResource.dispose] calls still execute.
  ///
  /// It is an error (throws [StateError]) if this scope has already exited.
  void defer(void Function() callback) {
    if (_isClosed) {
      throw StateError('Cannot register a defer callback in a closed scope.');
    }
    (_defers ??= []).add(callback);
  }

  /// Detaches [resource] from this scope and promotes it to the parent outer
  /// scope (or to manual ownership if there is no outer scope), returning
  /// [resource].
  ///
  /// Delegates to [ScopedResource.detachToParentScope] so resource-specific
  /// invariants (such as rejecting non-owning views) are enforced.
  ///
  /// It is an error (throws [StateError]) if this scope has already exited.
  T escape<T extends ScopedResource>(T resource) {
    if (_isClosed || _isDisposing) {
      throw StateError('Cannot escape a ScopedResource from a closed scope.');
    }
    var current = Zone.current[_scopeKey] as ResourceScope?;
    if (identical(current, this)) {
      return resource.detachToParentScope() as T;
    }
    final chain = <ResourceScope>[];
    while (current != null && !identical(current, this)) {
      chain.add(current);
      current = current._parentScope;
    }
    if (identical(current, this)) {
      chain.add(this);
      var result = resource;
      for (final s in chain) {
        result = runZoned(
          () => result.detachToParentScope() as T,
          zoneValues: {_scopeKey: s},
        );
      }
      return result;
    }
    return runZoned(
      () => resource.detachToParentScope() as T,
      zoneValues: {_scopeKey: this},
    );
  }

  void _track(ScopedResource resource) {
    if (_isClosed) {
      if (!resource.isDisposed) {
        resource.dispose();
      }
      throw StateError(
        'Cannot allocate or track a ScopedResource in a closed scope.',
      );
    }

    if (_set != null) {
      _set!.add(resource);
      return;
    }

    _list.add(resource);
    if (_list.length > 100) {
      _set = HashSet(equals: identical, hashCode: identityHashCode);
      _set!.addAll(_list);
      _list.clear();
    }
  }

  bool _untrack(ScopedResource resource) {
    if (_isClosed) return false;

    if (_set != null) {
      return _set!.remove(resource);
    }

    // O(1) swap-and-pop removal of all occurrences in flat list
    var removed = false;
    var i = 0;
    while (i < _list.length) {
      if (identical(_list[i], resource)) {
        final lastIdx = _list.length - 1;
        if (i < lastIdx) {
          _list[i] = _list[lastIdx];
        }
        _list.removeLast();
        removed = true;
      } else {
        i++;
      }
    }
    return removed;
  }

  void _dispose() {
    if (_isClosed || _isDisposing) return;
    _isDisposing = true;
    Object? firstError;
    StackTrace? firstStackTrace;

    final defers = _defers;
    if (defers != null) {
      while (defers.isNotEmpty) {
        final callback = defers.removeLast();
        try {
          callback();
        } catch (e, st) {
          firstError ??= e;
          firstStackTrace ??= st;
        }
      }
    }

    _isClosed = true;
    if (_set != null) {
      final resources = _set!.toList(growable: false);
      _set!.clear();
      for (final resource in resources) {
        if (!resource.isDisposed) {
          try {
            resource.dispose();
          } catch (e, st) {
            firstError ??= e;
            firstStackTrace ??= st;
          }
        }
      }
    } else {
      final resources = _list.toList(growable: false);
      _list.clear();
      for (final resource in resources) {
        if (!resource.isDisposed) {
          try {
            resource.dispose();
          } catch (e, st) {
            firstError ??= e;
            firstStackTrace ??= st;
          }
        }
      }
    }

    if (firstError != null) {
      Error.throwWithStackTrace(firstError, firstStackTrace!);
    }
  }
}
