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

import 'package:resource_scope/resource_scope.dart';

/// A simple custom resource implementing [ScopedResource] to represent
/// an externally allocated or native handle (e.g. native memory, socket, file).
final class SimpleNativeBuffer implements ScopedResource {
  final String name;
  bool _disposed = false;

  SimpleNativeBuffer(this.name) {
    print('Allocated: $name');
    // Automatically register this resource with the current ambient ResourceScope.
    ResourceScope.track(this);
  }

  @override
  bool get isDisposed => _disposed;

  @override
  void dispose() {
    if (!_disposed) {
      _disposed = true;
      print('Disposed: $name');
    }
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
  print('=== 1. Automatic Scoped Cleanup ===');
  ResourceScope.scope(() {
    final buffer1 = SimpleNativeBuffer('temp_buffer_1');
    final buffer2 = SimpleNativeBuffer('temp_buffer_2');
    print('Working with ${buffer1.name} and ${buffer2.name} inside scope...');
    // When the scope block exits, buffer1 and buffer2 are automatically disposed.
  });
  print('Scope exited.\n');

  print('=== 2. Promoting Results with ResourceScope.returning ===');
  final retained = ResourceScope.scope(() {
    // Intermediate buffers in the returning scope are disposed upon exit,
    // but the returned buffer is promoted to the caller scope.
    final result = ResourceScope.returning(() {
      final intermediate = SimpleNativeBuffer('intermediate_computation');
      final output = SimpleNativeBuffer('final_promoted_result');
      print(
        'Inside returning: created ${intermediate.name} and ${output.name}',
      );
      return output;
    });

    print('Output retained in outer scope: ${retainedResult(result)}');
    return result;
  });

  print('Outer scope completed. Final disposed status: ${retained.isDisposed}');
}

String retainedResult(SimpleNativeBuffer buf) =>
    '${buf.name} (disposed: ${buf.isDisposed})';
