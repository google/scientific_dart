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

import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Resource Management & Scoping', () {
    test('Explicit dispose releases GPU buffer memory', () {
      final arr = GpuArray.zeros([100, 100], DType.float64);
      expect(arr.isDisposed, isFalse);
      expect(arr.buffer.isDisposed, isFalse);

      arr.dispose();
      expect(arr.isDisposed, isTrue);
      expect(arr.buffer.isDisposed, isTrue);
    });

    test(
      'ResourceScope.scope automatically cleans up intermediate GPU allocations',
      () {
        GpuArray? outerRef;

        ResourceScope.scope(() {
          final a = GpuArray.ones([50, 50], DType.float64);
          final b = GpuArray.ones([50, 50], DType.float64);
          final c = a + b;
          outerRef = c;
          expect(outerRef!.isDisposed, isFalse);
        });

        // Scope has exited, everything should be disposed
        expect(outerRef!.isDisposed, isTrue);
      },
    );

    test(
      'ResourceScope.returning promotes returned GPU array to outer scope',
      () {
        GpuArray? intermediateRef;

        final result = ResourceScope.returning(() {
          final a = GpuArray.ones([10, 10], DType.float64);
          final b = GpuArray.ones([10, 10], DType.float64);
          intermediateRef = a;
          final res = a.matmul(b);
          return res;
        });

        // Intermediate array should be disposed
        expect(intermediateRef!.isDisposed, isTrue);

        // Returned array should remain active
        expect(result.isDisposed, isFalse);
        expect(result.shape, equals([10, 10]));
        result.dispose();
        expect(result.isDisposed, isTrue);
      },
    );
    test(
      'ResourceScope.returning promotes subviews without use-after-free',
      () {
        GpuArray? parentRef;

        final subview = ResourceScope.returning(() {
          final parent = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          parentRef = parent;
          final view = parent.slice([Slice(0, 2), Slice(1, 3)]);
          return view;
        });

        // Parent array is disposed at scope exit
        expect(parentRef!.isDisposed, isTrue);

        // Subview and its underlying buffer remain active and accessible
        expect(subview.isDisposed, isFalse);
        expect(subview.buffer.isDisposed, isFalse);
        expect(subview.shape, equals([2, 2]));
        expect(subview.toList(), equals([2.0, 3.0, 5.0, 6.0]));

        // Disposing the subview finally releases the buffer
        subview.dispose();
        expect(subview.isDisposed, isTrue);
        expect(subview.buffer.isDisposed, isTrue);
      },
    );

    test('Multiple views reference count buffer correctly', () {
      final base = GpuArray.fromList(
        [10.0, 20.0, 30.0, 40.0],
        [4],
        DType.float64,
      );
      final v1 = base.slice([Slice(0, 2)]);
      final v2 = base.slice([Slice(2, 4)]);
      expect(base.buffer.refCount, equals(3));

      base.dispose();
      expect(base.isDisposed, isTrue);
      expect(v1.isDisposed, isFalse);
      expect(v1.buffer.isDisposed, isFalse);
      expect(v1.toList(), equals([10.0, 20.0]));

      v1.dispose();
      expect(v1.isDisposed, isTrue);
      expect(v2.isDisposed, isFalse);
      expect(v2.buffer.isDisposed, isFalse);
      expect(v2.toList(), equals([30.0, 40.0]));

      v2.dispose();
      expect(v2.isDisposed, isTrue);
      expect(v2.buffer.isDisposed, isTrue);
    });

    test(
      'GpuDevice activeBufferCount and memory tracking with WeakReference',
      () {
        final dev = GpuDevice.create(name: 'Test Dev');
        expect(dev.activeBufferCount, equals(0));

        final buf1 = dev.createBuffer(
          sizeInBytes: 128,
          usage: GpuBufferUsage.storage,
        );
        final buf2 = dev.createBuffer(
          sizeInBytes: 256,
          usage: GpuBufferUsage.storage,
        );
        expect(dev.activeBufferCount, equals(2));
        expect(dev.allocatedMemoryBytes, equals(384));

        buf1.dispose();
        expect(dev.activeBufferCount, equals(1));
        expect(dev.allocatedMemoryBytes, equals(256));

        dev.dispose();
        expect(dev.isDisposed, isTrue);
        expect(buf2.isDisposed, isTrue);
      },
    );

    test('Accessing disposed GpuArray or GpuBuffer throws StateError', () {
      final arr = GpuArray.fromList([1.0, 2.0], [2], DType.float64);
      final buf = arr.buffer;
      arr.dispose();

      expect(arr.isDisposed, isTrue);
      expect(buf.isDisposed, isTrue);
      expect(() => arr.toList(), throwsA(isA<StateError>()));
      expect(() => buf.address, throwsA(isA<StateError>()));
    });

    test(
      'ResourceScope.returning with reshape and transpose views releases buffer on view dispose',
      () {
        GpuBuffer? underlyingBuffer;
        final transposed = ResourceScope.returning(() {
          final base = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );
          underlyingBuffer = base.buffer;
          final reshaped = base.reshape([2, 3]);
          return reshaped.transpose();
        });

        expect(transposed.isDisposed, isFalse);
        expect(underlyingBuffer!.isDisposed, isFalse);
        expect(transposed.shape, equals([3, 2]));
        expect(
          transposed.toNestedList(),
          equals([
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
          ]),
        );

        transposed.dispose();
        expect(transposed.isDisposed, isTrue);
        expect(underlyingBuffer!.isDisposed, isTrue);
      },
    );
  });
}
