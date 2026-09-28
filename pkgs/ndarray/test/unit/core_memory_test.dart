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

import 'dart:ffi' as ffi;
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Core Memory and Engine Fixes', () {
    group('ScratchArena Stack Allocation & Reentrancy', () {
      test('Nested getStridedBuffer calls do not clobber each other', () {
        final outerMarker = ScratchArena.marker;
        try {
          final outerBuf = ScratchArena.getStridedBuffer(4, 2); // 8 ints
          for (var i = 0; i < 8; i++) {
            outerBuf[i] = 100 + i;
          }

          // Nested scope
          final innerMarker = ScratchArena.marker;
          try {
            final innerBuf = ScratchArena.getStridedBuffer(4, 2);
            for (var i = 0; i < 8; i++) {
              innerBuf[i] = 200 + i;
            }

            // Verify inner has its own data
            for (var i = 0; i < 8; i++) {
              expect(innerBuf[i], equals(200 + i));
            }

            // Verify outer was not overwritten by inner allocation/writes
            for (var i = 0; i < 8; i++) {
              expect(outerBuf[i], equals(100 + i));
            }
          } finally {
            ScratchArena.reset(innerMarker);
          }

          // Outer buffer is still intact after inner reset
          for (var i = 0; i < 8; i++) {
            expect(outerBuf[i], equals(100 + i));
          }
        } finally {
          ScratchArena.reset(outerMarker);
        }
      });

      test('Zero-dim getStridedBuffer handles ndim=0 safely', () {
        final marker = ScratchArena.marker;
        try {
          final buf = ScratchArena.getStridedBuffer(0);
          expect(buf.address, isNot(equals(0)));
        } finally {
          ScratchArena.reset(marker);
        }
      });
    });

    group('NDArray.arange integer coercion & types', () {
      test('arange with Int32 dtype', () {
        final a = NDArray<Int32>.arange(0, 5, step: 1, dtype: DType.int32);
        expect(a.dtype, equals(DType.int32));
        expect(a.size, equals(5));
        expect(a.toList(), equals([0, 1, 2, 3, 4]));
        a.dispose();
      });

      test('arange with Int64 dtype and custom step', () {
        final a = NDArray<Int64>.arange(10, 20, step: 2, dtype: DType.int64);
        expect(a.dtype, equals(DType.int64));
        expect(a.size, equals(5));
        expect(a.toList(), equals([10, 12, 14, 16, 18]));
        a.dispose();
      });

      test('arange with Uint8 dtype', () {
        final a = NDArray<Uint8>.arange(0, 4, dtype: DType.uint8);
        expect(a.dtype, equals(DType.uint8));
        expect(a.size, equals(4));
        expect(a.toList(), equals([0, 1, 2, 3]));
        a.dispose();
      });

      test('arange with Int16 dtype and negative step', () {
        final a = NDArray<Int16>.arange(5, 0, step: -1, dtype: DType.int16);
        expect(a.dtype, equals(DType.int16));
        expect(a.size, equals(5));
        expect(a.toList(), equals([5, 4, 3, 2, 1]));
        a.dispose();
      });

      test('arange with Complex128 dtype', () {
        final a = NDArray<Complex128>.arange(0, 3, dtype: DType.complex128);
        expect(a.dtype, equals(DType.complex128));
        expect(a.size, equals(3));
        expect(
          a.toList(),
          equals([Complex(0, 0), Complex(1, 0), Complex(2, 0)]),
        );
        a.dispose();
      });

      test('arange with Complex64 dtype', () {
        final a = NDArray<Complex64>.arange(1, 3, dtype: DType.complex64);
        expect(a.dtype, equals(DType.complex64));
        expect(a.size, equals(2));
        expect(a.toList(), equals([Complex(1, 0), Complex(2, 0)]));
        a.dispose();
      });
    });

    group('NDArray.view empty dimensions & size calculation', () {
      test('View with 0-sized dimension [0]', () {
        final parent = NDArray<Float64>.zeros([10], DType.float64);
        final view = NDArray<Float64>.view(parent, shape: [0], strides: [1]);
        expect(view.size, equals(0));
        expect(view.shape, equals([0]));
        expect(view.toList(), isEmpty);
        parent.dispose();
      });

      test('View with 2D empty dimension [0, 5]', () {
        final parent = NDArray<Float64>.zeros([20], DType.float64);
        final view = NDArray<Float64>.view(
          parent,
          shape: [0, 5],
          strides: [5, 1],
        );
        expect(view.size, equals(0));
        expect(view.shape, equals([0, 5]));
        expect(view.toList(), isEmpty);
        parent.dispose();
      });

      test('View with 3D empty middle dimension [2, 0, 3]', () {
        final parent = NDArray<Float64>.zeros([20], DType.float64);
        final view = NDArray<Float64>.view(
          parent,
          shape: [2, 0, 3],
          strides: [6, 3, 1],
        );
        expect(view.size, equals(0));
        expect(view.shape, equals([2, 0, 3]));
        expect(view.toList(), isEmpty);
        parent.dispose();
      });

      test('0D scalar view []', () {
        final parent = NDArray<Float64>.fromList(
          [42.0, 99.0],
          [2],
          DType.float64,
        );
        final view = NDArray<Float64>.view(
          parent,
          shape: [],
          strides: [],
          offsetElements: 1,
        );
        expect(view.size, equals(1));
        expect(view.shape, equals(<int>[]));
        expect(view.scalar, equals(99.0));
        parent.dispose();
      });

      test('Empty slice on non-empty array', () {
        final parent = NDArray<Float64>.arange(0, 10, dtype: DType.float64);
        final sliced = parent[Slice(start: 3, stop: 3)];
        expect(sliced.size, equals(0));
        expect(sliced.shape, equals([0]));
        parent.dispose();
      });
    });

    group('NDArray contentHashCode and hashCode for Uint8 and Int16', () {
      test('Uint8 array contentHashCode computation', () {
        final a = NDArray<Uint8>.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
        final b = NDArray<Uint8>.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
        final c = NDArray<Uint8>.fromList([1, 2, 3, 5], [2, 2], DType.uint8);

        expect(a.contentHashCode, equals(b.contentHashCode));
        expect(a.contentHashCode, isNot(equals(c.contentHashCode)));
        expect(a.hashCode, equals(identityHashCode(a)));

        a.dispose();
        b.dispose();
        c.dispose();
      });

      test('Int16 array contentHashCode computation', () {
        final a = NDArray<Int16>.fromList(
          [100, 200, 300, 400],
          [2, 2],
          DType.int16,
        );
        final b = NDArray<Int16>.fromList(
          [100, 200, 300, 400],
          [2, 2],
          DType.int16,
        );
        final c = NDArray<Int16>.fromList(
          [100, 200, 300, 401],
          [2, 2],
          DType.int16,
        );

        expect(a.contentHashCode, equals(b.contentHashCode));
        expect(a.contentHashCode, isNot(equals(c.contentHashCode)));
        expect(a.hashCode, equals(identityHashCode(a)));

        a.dispose();
        b.dispose();
        c.dispose();
      });

      test('Strided non-contiguous Uint8 and Int16 contentHashCode', () {
        final a = NDArray<Uint8>.fromList(
          [1, 2, 3, 4, 5, 6],
          [2, 3],
          DType.uint8,
        );
        final aT = a.transpose();
        expect(aT.contentHashCode, isA<int>());

        final b = NDArray<Int16>.fromList(
          [10, 20, 30, 40, 50, 60],
          [2, 3],
          DType.int16,
        );
        final bT = b.transpose();
        expect(bT.contentHashCode, isA<int>());

        a.dispose();
        b.dispose();
      });

      test('Empty Uint8 and Int16 array contentHashCode', () {
        final emptyU8 = NDArray<Uint8>.zeros([0], DType.uint8);
        final emptyI16 = NDArray<Int16>.zeros([0], DType.int16);

        expect(emptyU8.contentHashCode, isA<int>());
        expect(emptyI16.contentHashCode, isA<int>());

        emptyU8.dispose();
        emptyI16.dispose();
      });
    });
  });
}
