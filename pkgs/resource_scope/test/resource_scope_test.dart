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
import 'package:test/test.dart';

final class DummyResource implements ScopedResource {
  final int id;
  bool _disposed = false;

  DummyResource(this.id) {
    ResourceScope.track(this);
  }

  @override
  bool get isDisposed => _disposed;

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
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
  group('ResourceScope', () {
    test('automatic disposal of resources in scope', () {
      DummyResource? r1;
      DummyResource? r2;

      ResourceScope.scope(() {
        r1 = DummyResource(1);
        r2 = DummyResource(2);
        expect(r1!.isDisposed, isFalse);
        expect(r2!.isDisposed, isFalse);
      });

      expect(r1!.isDisposed, isTrue);
      expect(r2!.isDisposed, isTrue);
    });

    test('returning resource promotes it out of scope', () {
      DummyResource? innerRes;

      final survived = ResourceScope.returning(() {
        innerRes = DummyResource(10);
        final temp = DummyResource(11);
        expect(temp.isDisposed, isFalse);
        return innerRes!;
      });

      expect(survived.isDisposed, isFalse);
      expect(survived.id, 10);
      survived.dispose();
      expect(survived.isDisposed, isTrue);
    });

    test('large scope (>100 resources) scales to HashSet cleanly', () {
      final weakRefs = <DummyResource>[];

      ResourceScope.scope(() {
        for (var i = 0; i < 150; i++) {
          weakRefs.add(DummyResource(i));
        }
        for (final r in weakRefs) {
          expect(r.isDisposed, isFalse);
        }
      });

      expect(weakRefs.length, 150);
      for (final r in weakRefs) {
        expect(r.isDisposed, isTrue);
      }
    });

    test('unmanaged scope ignores tracking', () {
      DummyResource? r;
      ResourceScope.scope(() {
        ResourceScope.unmanaged(() {
          r = DummyResource(99);
        });
      });

      expect(r!.isDisposed, isFalse);
      r!.dispose();
      expect(r!.isDisposed, isTrue);
    });
  });
}
