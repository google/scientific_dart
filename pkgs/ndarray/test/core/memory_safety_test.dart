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
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Memory Safety Core Tests', () {
    group('Scope Tests', () {
      test('Scope automatically disposes arrays', () {
        late NDArray<Float64> a;
        late NDArray<Float64> b;
        late NDArray<Float64> c;

        NDArray.scope(() {
          a = NDArray.zeros([10], DType.float64);
          b = NDArray.ones([10], DType.float64);
          c = add(a, b);

          expect(a.isDisposed, isFalse);
          expect(b.isDisposed, isFalse);
          expect(c.isDisposed, isFalse);
        });

        expect(a.isDisposed, isTrue);
        expect(b.isDisposed, isTrue);
        expect(c.isDisposed, isTrue);
      });

      test('detachFromScope() preserves array after scope ends', () {
        late NDArray<Float64> a;
        late NDArray<Float64> result;

        result = NDArray.scope(() {
          a = NDArray.zeros([10], DType.float64);
          final b = NDArray.ones([10], DType.float64);
          final c = add(a, b);
          return c.detachFromScope();
        });

        expect(a.isDisposed, isTrue);
        expect(result.isDisposed, isFalse);

        // Clean up manually since it was detached
        result.dispose();
        expect(result.isDisposed, isTrue);
      });

      test('Nested scopes work correctly', () {
        late NDArray outer;
        late NDArray inner;

        NDArray.scope(() {
          outer = NDArray.zeros([10], DType.float64);

          NDArray.scope(() {
            inner = NDArray.ones([10], DType.float64);
            expect(inner.isDisposed, isFalse);
          });

          expect(inner.isDisposed, isTrue);
          expect(outer.isDisposed, isFalse);
        });

        expect(outer.isDisposed, isTrue);
      });

      test('Scope disposes on error', () {
        late NDArray a;

        try {
          NDArray.scope(() {
            a = NDArray.zeros([10], DType.float64);
            throw Exception('test error');
          });
        } catch (_) {
          // expected
        }

        expect(a.isDisposed, isTrue);
      });

      test('Nested scope detach completely removes array from all scopes', () {
        late NDArray outer;
        late NDArray inner;

        NDArray.scope(() {
          outer = NDArray.zeros([10], DType.float64);

          NDArray.scope(() {
            inner = NDArray.ones([10], DType.float64);
            inner.detachFromScope(); // Completely detached!
            expect(inner.isDisposed, isFalse);
          });

          // Inner scope ended, 'inner' remains alive
          expect(inner.isDisposed, isFalse);
          expect(outer.isDisposed, isFalse);
        });

        // Outer scope ended, 'outer' is disposed, but 'inner' is completely unmanaged, so it survives!
        expect(outer.isDisposed, isTrue);
        expect(inner.isDisposed, isFalse);

        // Clean up manually
        inner.dispose();
        expect(inner.isDisposed, isTrue);
      });

      test('Hybrid List-to-Set promotion triggers past 100 arrays', () {
        final trackedArrays = <NDArray>[];

        NDArray.scope(() {
          // Allocate 105 arrays to force the List-to-Set threshold promotion crossing!
          for (var i = 0; i < 105; i++) {
            final a = NDArray.zeros([1], DType.float64);
            trackedArrays.add(a);
          }

          // Verify all are still alive
          for (final a in trackedArrays) {
            expect(a.isDisposed, isFalse);
          }
        });

        // Scope ended, so all 105 arrays should be automatically disposed of by the promoted HashSet!
        for (final a in trackedArrays) {
          expect(a.isDisposed, isTrue);
        }
      });

      test(
        'Nested scope detachToParentScope() promotes array to parent scope',
        () {
          late NDArray outer;
          late NDArray inner;

          NDArray.scope(() {
            outer = NDArray.zeros([10], DType.float64);

            NDArray.scope(() {
              inner = NDArray.ones([10], DType.float64);
              inner.detachToParentScope(); // Promoted to outer scope!
              expect(inner.isDisposed, isFalse);
            });

            // Inner scope ended, but 'inner' was promoted to outer scope, so it must still be alive!
            expect(inner.isDisposed, isFalse);
            expect(outer.isDisposed, isFalse);
          });

          // Outer scope ended, so both should now be disposed!
          expect(inner.isDisposed, isTrue);
          expect(outer.isDisposed, isTrue);
        },
      );

      test(
        'NDArray.unmanaged constructs arrays independent of current scope',
        () {
          late NDArray outer;
          late NDArray unmanagedArr;

          NDArray.scope(() {
            outer = NDArray.zeros([10], DType.float64);

            NDArray.unmanaged(() {
              unmanagedArr = NDArray.ones([10], DType.float64);
            });

            expect(outer.isDisposed, isFalse);
            expect(unmanagedArr.isDisposed, isFalse);
          });

          // Scope ended: 'outer' must be automatically disposed, but 'unmanagedArr' must survive completely!
          expect(outer.isDisposed, isTrue);
          expect(unmanagedArr.isDisposed, isFalse);

          // Clean up manually
          unmanagedArr.dispose();
          expect(unmanagedArr.isDisposed, isTrue);
        },
      );

      test(
        'NDArray.returning() returns result and disposes intermediate arrays',
        () {
          late NDArray<Float64> a;
          late NDArray<Float64> b;
          late NDArray<Float64> c;

          final result = NDArray.returning<Float64>(() {
            a = NDArray<Float64>.zeros([10], DType.float64);
            b = NDArray<Float64>.ones([10], DType.float64);
            c = add<Float64>(a, b);
            return c;
          });

          expect(a.isDisposed, isTrue);
          expect(b.isDisposed, isTrue);
          expect(result.isDisposed, isFalse);
          expect(result.toList(), List.filled(10, 1.0));

          result.dispose();
          expect(result.isDisposed, isTrue);
        },
      );

      test('NDArray.returning() promotes to outer scope when nested', () {
        late NDArray<Float64> innerRes;

        NDArray.scope(() {
          innerRes = NDArray.returning<Float64>(() {
            final temp = NDArray<Float64>.zeros([5], DType.float64);
            final ones = NDArray<Float64>.ones([5], DType.float64);
            return add<Float64>(temp, ones);
          });

          // Inside outer scope, innerRes is alive because it was promoted to outer scope
          expect(innerRes.isDisposed, isFalse);
        });

        // When outer scope exits, innerRes is disposed
        expect(innerRes.isDisposed, isTrue);
      });

      test(
        'NDArray.returning() disposes intermediate allocations on exception',
        () {
          late NDArray a;
          late NDArray b;

          expect(() {
            NDArray.returning<Float64>(() {
              a = NDArray.zeros([10], DType.float64);
              b = NDArray.ones([10], DType.float64);
              throw StateError('Intentional error inside returning callback');
            });
          }, throwsStateError);

          expect(a.isDisposed, isTrue);
          expect(b.isDisposed, isTrue);
        },
      );
    });

    group('ScratchArena Tests', () {
      test('standard allocation and 8-byte alignment', () {
        final marker = ScratchArena.marker;
        try {
          final ptr1 = ScratchArena.allocate<ffi.Int32>(
            5,
          ); // 5 bytes -> aligned to 8
          final ptr2 = ScratchArena.allocate<ffi.Int32>(
            3,
          ); // 3 bytes -> aligned to 8

          // Verify alignment and non-overlapping addresses
          expect(ptr1.address % 8, equals(0));
          expect(ptr2.address % 8, equals(0));
          expect(ptr2.address - ptr1.address, equals(8));
        } finally {
          ScratchArena.reset(marker);
        }
      });

      test('stack rewind and marker resets within a page', () {
        final markerStart = ScratchArena.marker;
        final ptr1 = ScratchArena.allocate<ffi.Int32>(16);
        final markerMid = ScratchArena.marker;
        final ptr2 = ScratchArena.allocate<ffi.Int32>(16);

        // Reset back to mid-marker
        ScratchArena.reset(markerMid);
        final ptr3 = ScratchArena.allocate<ffi.Int32>(16);

        // Since we rewound, ptr3 should reuse ptr2's address space exactly!
        expect(ptr3.address, equals(ptr2.address));

        // Reset back to start
        ScratchArena.reset(markerStart);
        final ptr4 = ScratchArena.allocate<ffi.Int32>(16);
        expect(ptr4.address, equals(ptr1.address));
      });

      test('multi-page switching and stable address pointers', () {
        final marker = ScratchArena.marker;
        try {
          final ptr0 = ScratchArena.allocate<ffi.Uint8>(10);

          // Allocate enough to spill beyond the 256KB base page capacity
          // Page 0 is 256KB (262,144 bytes). We allocate 200KB twice.
          final ptrSpill1 = ScratchArena.allocate<ffi.Uint8>(200 * 1024);

          final markerSpill = ScratchArena.marker;
          final ptrSpill2 = ScratchArena.allocate<ffi.Uint8>(200 * 1024);

          // ptrSpill2 must have spilled over to Page 1!
          // We verify both ptr0 and ptrSpill1 remain 100% valid at stable addresses!
          expect(ptr0.address, isNot(equals(ffi.nullptr.address)));
          expect(ptrSpill1.address, isNot(equals(ptrSpill2.address)));
          expect(ptrSpill2.address, isNot(equals(ffi.nullptr.address)));

          // Resetting back to markerSpill and allocating again should work perfectly
          ScratchArena.reset(markerSpill);
          final ptrSpill3 = ScratchArena.allocate<ffi.Uint8>(200 * 1024);
          expect(ptrSpill3.address, equals(ptrSpill2.address));
        } finally {
          ScratchArena.reset(marker);
        }
      });

      test('dynamic custom page sizing for large requests', () {
        final marker = ScratchArena.marker;
        try {
          // Allocate a single massive buffer exceeding 2MB (Page 0 is 256KB, standard subsequent is 512KB)
          // Request exactly 3MB. It must allocate a custom 3MB page!
          final massiveSize = 3 * 1024 * 1024;
          final ptrMassive = ScratchArena.allocate<ffi.Uint8>(massiveSize);

          expect(ptrMassive.address, isNot(equals(ffi.nullptr.address)));
        } finally {
          ScratchArena.reset(marker);
        }
      });

      test('selective hybrid memory pruning on reset', () {
        final markerStart = ScratchArena.marker;

        // Allocate 4 sequential standard pages to expand the pool (256KB, 512KB, 1MB, 2MB etc.)
        // We do this by requesting slightly larger allocations sequentially
        ScratchArena.allocate<ffi.Uint8>(200 * 1024); // page 0 (256KB)
        ScratchArena.allocate<ffi.Uint8>(
          400 * 1024,
        ); // spills to page 1 (512KB)
        ScratchArena.allocate<ffi.Uint8>(800 * 1024); // spills to page 2 (1MB)
        ScratchArena.allocate<ffi.Uint8>(1600 * 1024); // spills to page 3 (2MB)

        // Resetting back to start (Page index 0) must trigger selective pruning!
        // It must deallocate excess pages but preserve the core pool (Page 0 and Page 1).
        ScratchArena.reset(markerStart);

        // We test that subsequent standard allocations reuse cached page 1 immediately
        // without crashes, and the total cached pool is preserved safely.
        final ptrRecycled = ScratchArena.allocate<ffi.Uint8>(200 * 1024);
        expect(ptrRecycled.address, isNot(equals(ffi.nullptr.address)));
      });

      test('small page pool shifting instead of freeing', () {
        final markerStart = ScratchArena.marker;

        // 1. Force allocation of Page 0 (256KB) and Page 1 (512KB)
        ScratchArena.allocate<ffi.Uint8>(200 * 1024); // page 0
        final markerMid = ScratchArena.marker;
        ScratchArena.allocate<ffi.Uint8>(400 * 1024); // page 1 (512KB)

        // 2. Rewind back to Page 1 (markerMid)
        ScratchArena.reset(markerMid);

        // 3. Allocate a custom 3MB page. Page 1 (512KB) is too small!
        // The pool-shifting logic must shift the 512KB page to the end of the pool
        // and insert the 3MB page at current index 1.
        final ptrCustom = ScratchArena.allocate<ffi.Uint8>(3 * 1024 * 1024);
        expect(ptrCustom.address, isNot(equals(ffi.nullptr.address)));

        ScratchArena.reset(markerStart);
      });

      test('copy list helper methods for all dtypes', () {
        final marker = ScratchArena.marker;
        try {
          // Test copyInts
          final ints = [10, -20, 30, 40, -50];
          final pInts = ScratchArena.copyInts(ints);
          expect(pInts.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < ints.length; i++) {
            expect(pInts[i], equals(ints[i]));
          }

          // Test copyInt32s
          final pInt32s = ScratchArena.copyInt32s(ints);
          expect(pInt32s.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < ints.length; i++) {
            expect(pInt32s[i], equals(ints[i]));
          }

          // Test copyInt64s
          final pInt64s = ScratchArena.copyInt64s(ints);
          expect(pInt64s.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < ints.length; i++) {
            expect(pInt64s[i], equals(ints[i]));
          }

          // Test copyDoubles
          final doubles = [1.25, -2.5, 3.75, 4.0];
          final pDoubles = ScratchArena.copyDoubles(doubles);
          expect(pDoubles.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < doubles.length; i++) {
            expect(pDoubles[i], equals(doubles[i]));
          }

          // Test copyFloats
          final pFloats = ScratchArena.copyFloats(doubles);
          expect(pFloats.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < doubles.length; i++) {
            expect(pFloats[i], closeTo(doubles[i], 1e-5));
          }

          // Test copyComplexes (regular list)
          final complexes = [Complex(1.0, 2.0), Complex(-3.0, 4.5)];
          final pComplexes = ScratchArena.copyComplexes(complexes);
          expect(pComplexes.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < complexes.length; i++) {
            expect(pComplexes[i * 2], equals(complexes[i].real));
            expect(pComplexes[i * 2 + 1], equals(complexes[i].imag));
          }

          // Test copyComplexes (ComplexList)
          final complexList = ComplexList([1.0, 2.0, -3.0, 4.5]);
          final pComplexList = ScratchArena.copyComplexes(complexList);
          expect(pComplexList.address % 8, equals(0));
          for (var i = 0; i < complexList.length; i++) {
            expect(pComplexList[i * 2], equals(complexList[i].real));
            expect(pComplexList[i * 2 + 1], equals(complexList[i].imag));
          }

          // Test copyFloatComplexes (regular list)
          final pFloatComplexes = ScratchArena.copyFloatComplexes(complexes);
          expect(pFloatComplexes.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < complexes.length; i++) {
            expect(pFloatComplexes[i * 2], closeTo(complexes[i].real, 1e-5));
            expect(
              pFloatComplexes[i * 2 + 1],
              closeTo(complexes[i].imag, 1e-5),
            );
          }

          // Test copyFloatComplexes (ComplexList)
          final pFloatComplexList = ScratchArena.copyFloatComplexes(
            complexList,
          );
          expect(pFloatComplexList.address % 8, equals(0));
          for (var i = 0; i < complexList.length; i++) {
            expect(
              pFloatComplexList[i * 2],
              closeTo(complexList[i].real, 1e-5),
            );
            expect(
              pFloatComplexList[i * 2 + 1],
              closeTo(complexList[i].imag, 1e-5),
            );
          }

          // Test copyBools
          final bools = [true, false, true, true, false];
          final pBools = ScratchArena.copyBools(bools);
          expect(pBools.address % 8, equals(0)); // Alignment check
          for (var i = 0; i < bools.length; i++) {
            expect(pBools[i] != 0, equals(bools[i]));
          }
        } finally {
          ScratchArena.reset(marker);
        }
      });

      test('reset throws StateError on stale or out-of-order marker', () {
        final markerStart = ScratchArena.marker;
        ScratchArena.allocate<ffi.Uint8>(128);
        final markerMid = ScratchArena.marker;
        ScratchArena.allocate<ffi.Uint8>(128);
        final markerAhead = ScratchArena.marker;

        // Reset back to markerMid
        ScratchArena.reset(markerMid);

        // Attempting to reset to markerAhead (which was allocated after markerMid and is now ahead of current offset) throws StateError
        expect(() => ScratchArena.reset(markerAhead), throwsStateError);

        // Reset all the way back to markerStart
        ScratchArena.reset(markerStart);

        // Attempting to reset to markerMid or markerAhead (both stale/ahead of markerStart) throws StateError
        expect(() => ScratchArena.reset(markerMid), throwsStateError);
        expect(() => ScratchArena.reset(markerAhead), throwsStateError);
      });
    });

    group('Pointer Safety Tests', () {
      test('Negative strides pointer offsetting', () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        // Slice with negative stride: [4.0, 3.0, 2.0, 1.0]
        final view = a.slice([Slice(step: -1)]);

        expect(view.toList(), [4.0, 3.0, 2.0, 1.0]);
        // Local offset model: view.data starts at physical 0 (since minPhysicalOffset is 0).
        // Logical start is physical 3.
        // So offset elements relative to data start is 3.
        expect(view.offsetElements, 3);

        // The logical start is at index 3.
        // So view.pointer should be parent.pointer + 3 * 8 bytes.
        final expectedAddress = a.pointer.address + 3 * 8;
        expect(view.pointer.address, expectedAddress);
      });

      test('Nested view offset accumulation', () {
        final a = NDArray.fromList(
          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
          [8],
          DType.float64,
        );
        // View 1: slice [2:7] -> [2.0, 3.0, 4.0, 5.0, 6.0]
        final v1 = a.slice([Slice(start: 2, stop: 7)]);
        expect(v1.pointer.address, a.pointer.address + 2 * 8);
        expect(v1.offsetElements, 0); // starts at data start (physical 2)

        // View 2: slice of v1 with negative stride [::-1] -> [6.0, 5.0, 4.0, 3.0, 2.0]
        // Logical start of v2 is physical index 6 of root.
        // Physical range is [2, 6], so v2.data starts at physical 2.
        // Logical start (6) is index 4 relative to v2.data start (2).
        final v2 = v1.slice([Slice(step: -1)]);
        expect(v2.offsetElements, 4);
        expect(v2.pointer.address, a.pointer.address + 6 * 8);

        // View 3: slice of v2 with positive stride [1:4] -> [5.0, 4.0, 3.0]
        // v2 indices: 1, 2, 3 -> physical indices: 5, 4, 3.
        // Logical start of v3 is physical 5.
        // Physical range is [3, 5], so v3.data starts at physical 3.
        // Logical start (5) is index 2 relative to v3.data start (3).
        final v3 = v2.slice([Slice(start: 1, stop: 4)]);
        expect(v3.offsetElements, 2);
        expect(v3.pointer.address, a.pointer.address + 5 * 8);
        expect(v3.toList(), [5.0, 4.0, 3.0]);
      });

      test('min() along axis on negative-stride view fallback path', () {
        final a = NDArray.fromList(
          [10.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );

        // Slice to get a negative stride view along axis 1:
        // [3.0, 2.0, 10.0]
        // [6.0, 5.0,  4.0]
        final view = a.slice([Slice(), Slice(step: -1)]);
        expect(view.toList(), [3.0, 2.0, 10.0, 6.0, 5.0, 4.0]);

        // Now call min along axis 0.
        // Expected result: [min(3,6), min(2,5), min(10,4)] = [3.0, 2.0, 4.0]
        final res = min(view, axis: 0);
        expect(res.toList(), [3.0, 2.0, 4.0]);
      });

      test('cumprod() fallback on negative-stride view', () {
        // cumprod uses cumOpFFI.
        // For uint8, it uses fallback.
        final a = NDArray.fromList([1, 2, 3, 4], [4], DType.uint8);
        final view = a.slice([Slice(step: -1)]); // [4, 3, 2, 1]
        expect(view.toList(), [4, 3, 2, 1]);

        // cumprod along axis 0
        final res = cumprod(view, axis: 0);
        expect(res.toList(), [4, 12, 24, 24]);
      });

      test('sqrt() fallback on negative-stride view', () {
        // sqrt fallback for non-contiguous uses copy, which we optimized.
        final a = NDArray.fromList([4, 9, 16, 25], [4], DType.int32);
        final view = a.slice([Slice(step: -1)]); // [25, 16, 9, 4]
        expect(view.toList(), [25, 16, 9, 4]);

        final res = sqrt(view);
        expect(res.toList(), [5.0, 4.0, 3.0, 2.0]);
      });

      test('out buffer contiguity assertion', () {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);

        // Create a non-contiguous out buffer
        final parent = NDArray.create([4], DType.float64);
        final nonContigOut = parent.slice([
          Slice(start: 0, stop: 4, step: 2),
        ]); // shape [2], non-contiguous

        expect(() => matmul(a, b, out: nonContigOut), throwsArgumentError);
      });
    });

    group('External Memory Tests', () {
      test('Wrap raw pointer and access data (Externally Managed)', () {
        final pointer = malloc<ffi.Double>(4);
        for (var i = 0; i < 4; i++) {
          pointer[i] = (i + 1) * 10.0;
        }

        final arr = NDArray<AnySpec>.fromPointer(pointer.cast(), [
          2,
          2,
        ], DType.float64);

        expect(arr.shape, [2, 2]);
        expect(arr.data, [10.0, 20.0, 30.0, 40.0]);

        // Modify through NDArray
        arr.data[1] = 99.0;
        expect(pointer[1], 99.0);

        // Slicing
        final slice = arr.slice([Slice.all(), Slice(start: 1)]);
        expect(slice.shape, [2, 1]);
        expect(slice.toList(), [99.0, 40.0]);

        expect(arr.isDisposed, false);
        arr.dispose();
        expect(arr.isDisposed, true);
        expect(slice.isDisposed, true);

        // Ensure calling dispose on externally managed pointer array does not free the raw pointer
        expect(pointer[0], 10.0); // Still accessible since it wasn't freed!

        malloc.free(pointer);
      });

      test(
        'Wrap with Custom Native Finalizer and manually dispose',
        () {
          final pointer = malloc<ffi.Double>(4);
          for (var i = 0; i < 4; i++) {
            pointer[i] = (i + 1) * 2.0;
          }

          final arr = NDArray<AnySpec>.fromPointer(
            pointer.cast(),
            [4],
            DType.float64,
            nativeFinalizer: malloc.nativeFree.cast(),
          );

          expect(arr.data, [2.0, 4.0, 6.0, 8.0]);
          expect(arr.isDisposed, false);

          // Manual dispose should invoke custom finalizer and free memory
          arr.dispose();
          expect(arr.isDisposed, true);

          // Note: The backing memory pointer is now freed, accessing it is undefined behavior.
        },
        skip: const bool.fromEnvironment('dart.tool.dart2wasm')
            ? 'NativeFinalizer / Native.addressOf is not supported on Wasm'
            : false,
      );

      test(
        'Custom native finalizers are rejected on Wasm',
        () {
          final pointer = malloc<ffi.Double>(4);
          try {
            expect(
              () => NDArray<AnySpec>.fromPointer(
                pointer.cast(),
                [4],
                DType.float64,
                nativeFinalizer: ffi.Pointer.fromAddress(1).cast(),
              ),
              throwsUnsupportedError,
            );
          } finally {
            malloc.free(pointer);
          }
        },
        skip: const bool.fromEnvironment('dart.tool.dart2wasm')
            ? false
            : 'Only dart2wasm rejects custom native finalizers',
      );

      test('Integration with NDArray.scope', () {
        final pointer = malloc<ffi.Double>(2);
        pointer[0] = 1.0;
        pointer[1] = 2.0;

        NDArray<AnySpec>? arrRef;

        NDArray.scope(() {
          final arr = NDArray<AnySpec>.fromPointer(pointer.cast(), [
            2,
          ], DType.float64);
          arrRef = arr;
          expect(arr.isDisposed, false);
        });

        // Once scope exits, the pointer array is marked as disposed to protect against invalid access
        expect(arrRef!.isDisposed, true);
        expect(() => arrRef!.data, throwsStateError);

        // Since it was externally managed, the raw pointer was not freed by the scope
        expect(pointer[0], 1.0);
        malloc.free(pointer);
      });
    });

    group('Code Safety Preconditions Tests', () {
      test('spacers numSamples non-negative check', () {
        NDArray.scope(() {
          // logspace and geomspace return empty for numSamples == 0 and throw for negative
          expect(logspace(0.0, 3.0, 0, dtype: DType.float64).size, equals(0));
          expect(
            () => logspace(0.0, 3.0, -5, dtype: DType.float64),
            throwsArgumentError,
          );
          expect(
            geomspace(1.0, 100.0, 0, dtype: DType.float64).size,
            equals(0),
          );
          expect(
            () => geomspace(1.0, 100.0, -3, dtype: DType.float64),
            throwsArgumentError,
          );
        });
      });

      test('GridRange constructor parameter validation check', () {
        // GridRange must throw ArgumentError for zero step or non-positive numPoints
        expect(() => GridRange(0.0, 10.0, step: 0.0), throwsArgumentError);
        expect(() => GridRange(0.0, 10.0, numPoints: 0), throwsArgumentError);
        expect(() => GridRange(0.0, 10.0, numPoints: -5), throwsArgumentError);
      });

      test('FFT functions input disposed array state check', () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        a.dispose();

        // fft and ifft must throw StateError if input is disposed
        expect(() => fft(a), throwsStateError);
        expect(() => ifft(a), throwsStateError);
      });
    });
    test(
      'View Invalidation Safety (throws StateError on disposed views)',
      () => NDArray.scope(() {
        final parent = NDArray.ones([2, 2], DType.float64);
        final view = parent.slice([Index(0)]); // Select first row

        expect(parent.isDisposed, false);
        expect(view.isDisposed, false);

        parent.dispose();

        expect(parent.isDisposed, true);
        expect(view.isDisposed, true);

        // Accessing public APIs should throw StateError
        expect(() => view.toList(), throwsStateError);
        expect(() => view.pointer, throwsStateError);
        expect(() => view[0], throwsStateError);
        expect(() => view.reshape([2]), throwsStateError);
      }),
    );

    group(
      'Allocation Tracking Integration Tests',
      () {
        late File tempFile;
        late String workingDir;

        setUp(() {
          final inRoot = Directory('pkgs/ndarray').existsSync();
          workingDir = inRoot ? 'pkgs/ndarray' : '.';
          final dirPath = inRoot ? 'pkgs/ndarray/test/core' : 'test/core';
          tempFile = File('$dirPath/temp_tracker_helper.dart');
          tempFile.writeAsStringSync('''
import 'package:ndarray/ndarray.dart';

void main() {
  print('trackAllocations: \${NDArray.trackAllocations}');
  final a = NDArray.zeros([10], DType.float64);
  print('trackedCount: \${NDArray.trackedAllocations.length}');
  
  // Test views are not tracked directly
  a.slice([Index(0)]);
  print('trackedCountAfterView: \${NDArray.trackedAllocations.length}');

  // Test checkNoLeaks throws
  try {
    NDArray.checkNoLeaks();
    print('checkNoLeaks: OK');
  } catch (e) {
    print('checkNoLeaks: FAILED');
  }

  a.dispose();
  print('trackedCountAfterDispose: \${NDArray.trackedAllocations.length}');
  
  // Test checkNoLeaks passes
  try {
    NDArray.checkNoLeaks();
    print('checkNoLeaksAfterDispose: OK');
  } catch (e) {
    print('checkNoLeaksAfterDispose: FAILED');
  }
}
''');
        });

        tearDown(() {
          if (tempFile.existsSync()) {
            tempFile.deleteSync();
          }
        });

        test('Verifies tracking is disabled by default', () async {
          final result = await Process.run(Platform.resolvedExecutable, [
            tempFile.absolute.path,
          ], workingDirectory: workingDir);

          expect(result.exitCode, 0);
          expect(result.stdout, contains('trackAllocations: false'));
          expect(result.stdout, contains('trackedCount: 0'));
          expect(result.stdout, contains('trackedCountAfterView: 0'));
          expect(result.stdout, contains('checkNoLeaks: OK'));
          expect(result.stdout, contains('trackedCountAfterDispose: 0'));
          expect(result.stdout, contains('checkNoLeaksAfterDispose: OK'));
        });

        test('Verifies tracking works when enabled via define', () async {
          final result = await Process.run(Platform.resolvedExecutable, [
            '--define=TRACK_NDARRAY_ALLOCATIONS=true',
            tempFile.absolute.path,
          ], workingDirectory: workingDir);

          expect(result.exitCode, 0);
          expect(result.stdout, contains('trackAllocations: true'));
          expect(result.stdout, contains('trackedCount: 1'));
          expect(result.stdout, contains('trackedCountAfterView: 1'));
          expect(result.stdout, contains('checkNoLeaks: FAILED'));
          expect(result.stdout, contains('trackedCountAfterDispose: 0'));
          expect(result.stdout, contains('checkNoLeaksAfterDispose: OK'));
        });
      },
      skip: const bool.fromEnvironment('dart.tool.dart2wasm')
          ? 'Process.run is not supported on Wasm'
          : false,
    );

    group('Disposed Array Precondition Tests', () {
      test('distance operations throw StateError on disposed arrays', () {
        final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        x.dispose();
        expect(() => pdist(x), throwsStateError);

        final xa = NDArray.fromList([1.0, 2.0], [1, 2], DType.float64);
        final xb = NDArray.fromList([3.0, 4.0], [1, 2], DType.float64);
        xa.dispose();
        expect(() => cdist(xa, xb), throwsStateError);
      });

      test('financial operations throw StateError on disposed arrays', () {
        final rate = NDArray.scalar(0.05, dtype: DType.float64);
        final nper = NDArray.scalar(10.0, dtype: DType.float64);
        final pmt = NDArray.scalar(-100.0, dtype: DType.float64);
        final pvArr = NDArray.scalar(1000.0, dtype: DType.float64);
        final values = NDArray.fromList(
          [-100.0, 39.0, 59.0, 55.0, 20.0],
          [5],
          DType.float64,
        );

        rate.dispose();
        expect(() => fv(rate, nper, pmt, pvArr), throwsStateError);
        expect(() => pv(rate, nper, pmt, pvArr), throwsStateError);
        expect(() => npv(rate, values), throwsStateError);

        values.dispose();
        expect(() => irr(values), throwsStateError);
      });

      test('interpolation operation throws StateError on disposed arrays', () {
        final x = NDArray.fromList([2.5], [1], DType.float64);
        final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final fp = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);

        x.dispose();
        expect(() => interp(x, xp, fp), throwsStateError);
      });

      test('kron operation throws StateError on disposed arrays', () {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        a.dispose();
        expect(() => kron(a, b), throwsStateError);
      });

      test('kron disposes intermediate cast arrays within scope', () {
        late NDArray<Float64> result;
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final b = NDArray.fromList([0.5, 1.5], [2], DType.float64);
          expect(() => kron(a.asAnySpec, b.asAnySpec), throwsArgumentError);
          result = kron(a.astype(DType.float64), b);
          expect(result.isDisposed, isFalse);
          expect(result.dtype, DType.float64);
          expect(result.toList(), [0.5, 1.5, 1.0, 3.0]);
        });
        expect(result.isDisposed, isTrue);
      });
    });
  });

  group('Lifecycle boundary cases', () {
    test(
      'double dispose is a no-op and use-after-dispose throws StateError',
      () {
        final a = NDArray.ones([3, 3], DType.float64);
        a.dispose();
        expect(a.isDisposed, isTrue);
        // Double dispose is a safe no-op
        a.dispose();
        expect(() => a[[0, 0]], throwsStateError);
        expect(() => a.toList(), throwsStateError);
        expect(() => sum(a), throwsStateError);
        expect(() => sin(a), throwsStateError);
      },
    );

    test(
      'ScratchArena rejects negative sizes and out-of-order marker resets',
      () {
        expect(() => ScratchArena.allocate<ffi.Uint8>(-1), throwsArgumentError);
        expect(() => ScratchArena.getStridedBuffer(-1), throwsArgumentError);

        final mOuter = ScratchArena.marker;
        ScratchArena.allocate<ffi.Int64>(64);
        final mInner = ScratchArena.marker;
        ScratchArena.allocate<ffi.Int64>(64);

        // Resetting to outer first makes inner marker stale (ahead of current offset)
        ScratchArena.reset(mOuter);
        expect(() => ScratchArena.reset(mInner), throwsStateError);
      },
    );
  });
}
