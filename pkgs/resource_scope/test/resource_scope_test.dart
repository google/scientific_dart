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

import 'dart:async' show Zone;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

final class DummyResource implements ScopedResource {
  final int id;
  bool _disposed = false;

  DummyResource(this.id) {
    ResourceScope.track(this);
  }

  DummyResource.tracked(this.id, void Function(DummyResource) onCreated) {
    onCreated(this);
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

    test(
      'closed scope immediately disposes late tracked resource and throws StateError',
      () {
        late dynamic capturedZone;
        ResourceScope.scope(() {
          capturedZone = Zone.current;
        });

        DummyResource? constructed;
        expect(
          () {
            capturedZone.run(() {
              DummyResource.tracked(999, (r) => constructed = r);
            });
          },
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('closed scope'),
            ),
          ),
        );

        expect(constructed, isNotNull);
        expect(constructed!.isDisposed, isTrue);
      },
    );

    test('closed scope throws StateError when promoting resource', () {
      late dynamic capturedZone;
      late DummyResource unmanagedRes;
      ResourceScope.scope(() {
        capturedZone = Zone.current;
      });

      ResourceScope.unmanaged(() {
        unmanagedRes = DummyResource(43);
      });

      expect(
        () {
          capturedZone.run(() {
            ResourceScope.promoteToParent(unmanagedRes);
          });
        },
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('closed scope'),
          ),
        ),
      );

      unmanagedRes.dispose();
    });

    test(
      'detachToParentScope on outer-scope resource inside inner scope does not double-track',
      () {
        DummyResource? outerRes;
        ResourceScope.scope(() {
          outerRes = DummyResource(100);
          ResourceScope.scope(() {
            outerRes!.detachToParentScope();
          });
          expect(outerRes!.isDisposed, isFalse);
          // Detaching from outer scope should remove the single tracking entry.
          outerRes!.detachFromScope();
        });

        expect(outerRes!.isDisposed, isFalse);
        outerRes!.dispose();
        expect(outerRes!.isDisposed, isTrue);
      },
    );

    test(
      'detachToParentScope on unmanaged resource inside nested scopes does not hijack into outer scope',
      () {
        final unmanagedRes = DummyResource(200);
        ResourceScope.scope(() {
          ResourceScope.scope(() {
            unmanagedRes.detachToParentScope();
          });
          expect(unmanagedRes.isDisposed, isFalse);
        });

        expect(unmanagedRes.isDisposed, isFalse);
        unmanagedRes.dispose();
        expect(unmanagedRes.isDisposed, isTrue);
      },
    );

    test(
      'using supports LIFO defer callbacks and runs them before resource disposal',
      () {
        final events = <String>[];
        late DummyResource r1;
        late DummyResource r2;

        final (escaped1, escaped2) = ResourceScope.using((scope) {
          r1 = DummyResource(1);
          r2 = DummyResource(2);
          final out1 = DummyResource(3);
          final out2 = DummyResource(4);

          scope.defer(() {
            // Tracked resources should still be alive when defer callbacks run.
            expect(r1.isDisposed, isFalse);
            expect(r2.isDisposed, isFalse);
            events.add('defer-1');
          });
          scope.defer(() {
            expect(r1.isDisposed, isFalse);
            events.add('defer-2');
          });

          return (scope.escape(out1), scope.escape(out2));
        });

        expect(events, equals(['defer-2', 'defer-1']));
        expect(r1.isDisposed, isTrue);
        expect(r2.isDisposed, isTrue);
        expect(escaped1.isDisposed, isFalse);
        expect(escaped2.isDisposed, isFalse);

        escaped1.dispose();
        escaped2.dispose();
      },
    );

    test(
      'defer callbacks and resource disposals all run even when a defer throws',
      () {
        final events = <String>[];
        late DummyResource r1;

        expect(
          () => ResourceScope.using((scope) {
            r1 = DummyResource(10);
            scope.defer(() => events.add('defer-1'));
            scope.defer(() {
              events.add('defer-2-throws');
              throw StateError('boom in defer-2');
            });
            scope.defer(() => events.add('defer-3'));
          }),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'boom in defer-2',
            ),
          ),
        );

        expect(events, equals(['defer-3', 'defer-2-throws', 'defer-1']));
        expect(r1.isDisposed, isTrue);
      },
    );

    test(
      'async using runs defer in LIFO order after Future completes',
      () async {
        final events = <String>[];
        late DummyResource temp;

        final escaped = await ResourceScope.using((scope) async {
          temp = DummyResource(20);
          scope.defer(() => events.add('async-defer-1'));
          await Future<void>.delayed(Duration.zero);
          scope.defer(() => events.add('async-defer-2'));
          final out = DummyResource(21);
          return scope.escape(out);
        });

        expect(events, equals(['async-defer-2', 'async-defer-1']));
        expect(temp.isDisposed, isTrue);
        expect(escaped.isDisposed, isFalse);
        escaped.dispose();
      },
    );

    test('nested using promotes escaped resources to parent scope', () {
      late DummyResource innerTemp;
      late DummyResource promotedToOuter;

      ResourceScope.using((outer) {
        final res = ResourceScope.using((inner) {
          innerTemp = DummyResource(30);
          final out = DummyResource(31);
          return inner.escape(out);
        });
        promotedToOuter = res;
        expect(innerTemp.isDisposed, isTrue);
        expect(promotedToOuter.isDisposed, isFalse);
      });

      expect(promotedToOuter.isDisposed, isTrue);
    });

    test('calling defer or escape on a closed scope throws StateError', () {
      late ResourceScope capturedScope;
      ResourceScope.using((scope) {
        capturedScope = scope;
      });

      final r = DummyResource(40);
      expect(() => capturedScope.defer(() {}), throwsStateError);
      expect(() => capturedScope.escape(r), throwsStateError);
      r.dispose();
    });
  });
}
