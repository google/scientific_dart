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
/// Use [ResourceScope.scope] to execute code blocks where any created [ScopedResource]
/// is automatically disposed when the scope finishes.
///
/// Use [ResourceScope.returning] to automatically promote a returned [ScopedResource]
/// to the outer parent scope while disposing all intermediate allocations.
final class ResourceScope {
  ResourceScope._();

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
    final scope = Zone.current[_scopeKey] as _ResourceScopeInstance?;
    scope?._track(resource);
    if (trackAllocations) {
      _trackedAllocations.add(resource);
    }
  }

  /// Removes [resource] from the active [ResourceScope] in [Zone.current], if any.
  static void untrack(ScopedResource resource) {
    final scope = Zone.current[_scopeKey] as _ResourceScopeInstance?;
    scope?._untrack(resource);
    if (trackAllocations) {
      _trackedAllocations.remove(resource);
    }
  }

  /// Executes [callback] within an automatic resource management scope.
  ///
  /// Any [ScopedResource] registered via [track] during the execution of
  /// [callback] will be automatically disposed of when the callback finishes (or throws).
  ///
  /// Supports both synchronous callbacks and asynchronous [Future]-returning callbacks.
  static R scope<R>(R Function() callback) {
    final parentScope = Zone.current[_scopeKey] as _ResourceScopeInstance?;
    final scope = _ResourceScopeInstance(parentScope);
    return runZoned(() {
      R result;
      try {
        result = callback();
      } catch (e) {
        scope.dispose();
        rethrow;
      }

      if (result is Future) {
        return result.whenComplete(scope.dispose) as R;
      }
      scope.dispose();
      return result;
    }, zoneValues: {_scopeKey: scope});
  }

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
    final scope = Zone.current[_scopeKey] as _ResourceScopeInstance?;
    if (scope == null) {
      throw StateError(
        'detachToParentScope() is only valid inside an active NDArray scope.',
      );
    }
    if (scope._isClosed) {
      throw StateError('Cannot promote a ScopedResource from a closed scope.');
    }
    final wasTracked = scope._untrack(resource);
    if (wasTracked) {
      scope._parentScope?._track(resource);
    }
  }
}

/// Private zone scope instance that tracks [ScopedResource]s using a hybrid
/// collection: a flat [List] for small scopes (<= 100 resources) and a
/// [HashSet] for large scopes (> 100 resources) for O(1) scaling.
final class _ResourceScopeInstance {
  final _ResourceScopeInstance? _parentScope;
  final List<ScopedResource> _list = [];
  Set<ScopedResource>? _set;
  bool _isClosed = false;

  _ResourceScopeInstance(this._parentScope);

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

  void dispose() {
    _isClosed = true;
    if (_set != null) {
      final resources = _set!.toList(growable: false);
      _set!.clear();
      for (final resource in resources) {
        if (!resource.isDisposed) {
          resource.dispose();
        }
      }
    } else {
      final resources = _list.toList(growable: false);
      _list.clear();
      for (final resource in resources) {
        if (!resource.isDisposed) {
          resource.dispose();
        }
      }
    }
  }
}
