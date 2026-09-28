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

import 'dart:isolate';

import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('SendableNDArray', () {
    group('Copy Mode (toSendable / fromCopy)', () {
      test('transfers contiguous 1D array across Isolate.run', () async {
        await NDArray.scope(() async {
          final original = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );
          final sendable = original.toSendable();

          expect(sendable.isCopy, isTrue);
          expect(sendable.isBorrowed, isFalse);
          expect(sendable.isMaterialized, isFalse);
          expect(sendable.shape, equals([5]));
          expect(sendable.dtype, equals(DType.float64));

          final workerResult = await Isolate.run(() {
            return NDArray.scope(() {
              setNumThreads(1);
              final workerArr = sendable.materialize();
              workerArr[0] = 99.0;
              return (
                shape: workerArr.shape,
                dtype: workerArr.dtype,
                sum: sum(workerArr).scalar,
              );
            });
          });

          expect(workerResult.shape, equals([5]));
          expect(workerResult.dtype, equals(DType.float64));
          expect(workerResult.sum, equals(99.0 + 2.0 + 3.0 + 4.0 + 5.0));
          // Main isolate array must remain unmutated (isolated copy)
          expect(original.getCell([0]), equals(1.0));
        });
      });

      test('transfers 2D multi-dimensional array across Isolate.run', () async {
        await NDArray.scope(() async {
          final original = NDArray<Float32>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float32,
          );
          final sendable = original.toSendable();

          final returnedSendable = await Isolate.run(() {
            return NDArray.scope(() {
              setNumThreads(1);
              final workerArr = sendable.materialize();
              final doubled = multiply(
                workerArr,
                NDArray<Float32>.full([2, 3], 2.0, dtype: DType.float32),
              );
              return doubled.toSendable();
            });
          });

          final finalResult = returnedSendable.materialize();
          expect(finalResult.shape, equals([2, 3]));
          expect(finalResult.dtype, equals(DType.float32));
          expect(
            finalResult.toList(),
            equals([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
          );
        });
      });

      test(
        'transfers non-contiguous strided slice across Isolate.run',
        () async {
          await NDArray.scope(() async {
            // 4x4 matrix
            final full = NDArray<Int32>.fromList(List.generate(16, (i) => i), [
              4,
              4,
            ], DType.int32);

            // Sub-slice with striding
            final slice = full.slice([
              Slice(start: 0, stop: 4, step: 2), // rows 0, 2
              Slice(start: 1, stop: 4, step: 2), // cols 1, 3
            ]);
            expect(slice.isContiguous, isFalse);
            expect(slice.shape, equals([2, 2]));
            expect(slice.toList(), equals([1, 3, 9, 11]));

            final sendable = slice.toSendable();
            final received = await Isolate.run(() {
              return NDArray.scope(() {
                final workerArr = sendable.materialize();
                return (
                  isContiguous: workerArr.isContiguous,
                  shape: workerArr.shape,
                  dtype: workerArr.dtype,
                  list: workerArr.toList(),
                );
              });
            });

            expect(received.isContiguous, isTrue);
            expect(received.shape, equals([2, 2]));
            expect(received.dtype, equals(DType.int32));
            expect(received.list, equals([1, 3, 9, 11]));
          });
        },
      );

      test(
        'transfers transposed non-contiguous array across Isolate.run',
        () async {
          await NDArray.scope(() async {
            final full = NDArray<Int32>.fromList(List.generate(16, (i) => i), [
              4,
              4,
            ], DType.int32);
            final transposed = full.transposed;
            expect(transposed.isContiguous, isFalse);
            final transposedSendable = transposed.toSendable();

            final transposedList = await Isolate.run(() {
              return NDArray.scope(() {
                final workerArr = transposedSendable.materialize();
                return (
                  isContiguous: workerArr.isContiguous,
                  list: workerArr.toList(),
                );
              });
            });

            expect(transposedList.isContiguous, isTrue);
            expect(transposedList.list, equals(transposed.toList()));
          });
        },
      );

      test('transfers empty arrays (size 0)', () async {
        await NDArray.scope(() async {
          final empty = NDArray<Float64>.zeros([0, 5], DType.float64);
          final sendable = empty.toSendable();

          final receivedShape = await Isolate.run(() {
            return NDArray.scope(() {
              final workerArr = sendable.materialize();
              return workerArr.shape;
            });
          });

          expect(receivedShape, equals([0, 5]));
        });
      });

      test('transfers Int64 dtype across Isolate.run', () async {
        await NDArray.scope(() async {
          final i64 = NDArray<Int64>.fromList([100, 200], [2], DType.int64);
          final s64 = i64.toSendable();
          final r64 = await Isolate.run(() {
            return NDArray.scope(() => s64.materialize().toList());
          });
          expect(r64, equals([100, 200]));
        });
      });

      test('transfers Uint8 dtype across Isolate.run', () async {
        await NDArray.scope(() async {
          final u8 = NDArray<Uint8>.fromList([0, 127, 255], [3], DType.uint8);
          final su8 = u8.toSendable();
          final ru8 = await Isolate.run(() {
            return NDArray.scope(() => su8.materialize().toList());
          });
          expect(ru8, equals([0, 127, 255]));
        });
      });

      test('transfers boolean dtype across Isolate.run', () async {
        await NDArray.scope(() async {
          final b = NDArray<Boolean>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          final sb = b.toSendable();
          final rb = await Isolate.run(() {
            return NDArray.scope(() => sb.materialize().toList());
          });
          expect(rb, equals([true, false, true]));
        });
      });

      test('transfers Complex128 dtype across Isolate.run', () async {
        await NDArray.scope(() async {
          final c = NDArray<Complex128>.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            DType.complex128,
          );
          final sc = c.toSendable();
          final rc = await Isolate.run(() {
            return NDArray.scope(() {
              final arr = sc.materialize();
              final c0 = arr.getCell([0]);
              final c1 = arr.getCell([1]);
              return [
                [c0.real, c0.imag],
                [c1.real, c1.imag],
              ];
            });
          });
          expect(rc, [
            [1.0, 2.0],
            [3.0, 4.0],
          ]);
        });
      });

      test('throws StateError when attempting to transfer disposed array', () {
        final arr = NDArray<Float64>.ones([5], DType.float64);
        arr.dispose();
        expect(() => arr.toSendable(), throwsStateError);
        expect(() => SendableNDArray.fromCopy(arr), throwsStateError);
      });

      test('throws StateError when materializing copy-mode twice', () {
        final arr = NDArray<Float64>.ones([5], DType.float64);
        final sendable = arr.toSendable();
        final first = sendable.materialize();
        expect(sendable.isMaterialized, isTrue);
        expect(() => sendable.materialize(), throwsStateError);
        first.dispose();
        arr.dispose();
      });

      test(
        'throws StateError when calling materializeView on copy-mode array',
        () {
          final arr = NDArray<Float64>.ones([5], DType.float64);
          final sendable = arr.toSendable();
          expect(() => sendable.materializeView(), throwsStateError);
          expect(() => sendable.address, throwsStateError);
          expect(() => sendable.physicalByteCapacity, throwsStateError);
          arr.dispose();
        },
      );
    });

    group('Borrow Mode (toSendableBorrow / unsafeBorrow)', () {
      test(
        'mutates full array in-place across Isolate.run without copies',
        () async {
          await NDArray.scope(() async {
            final array = NDArray<Float64>.zeros([10], DType.float64);
            final sendable = array.toSendableBorrow();

            expect(sendable.isBorrowed, isTrue);
            expect(sendable.isCopy, isFalse);
            expect(sendable.address, equals(array.pointer.address));
            expect(
              sendable.physicalByteCapacity,
              equals(array.physicalByteCapacity),
            );
            expect(sendable.shape, equals([10]));
            expect(sendable.dtype, equals(DType.float64));

            await Isolate.run(() {
              return NDArray.scope(() {
                setNumThreads(1);
                final view = sendable.materializeView();
                for (var i = 0; i < 10; i++) {
                  view[i] = (i + 1) * 7.0;
                }
              });
            });

            // Main isolate immediately observes the in-place mutations
            expect(
              array.toList(),
              equals([
                7.0,
                14.0,
                21.0,
                28.0,
                35.0,
                42.0,
                49.0,
                56.0,
                63.0,
                70.0,
              ]),
            );
          });
        },
      );

      test('mutates an output slice in-place across Isolate.run', () async {
        await NDArray.scope(() async {
          // 4x4 matrix initialized to zeros
          final matrix = NDArray<Float64>.zeros([4, 4], DType.float64);

          // Take an interior sub-slice (rows 1..3, cols 1..3) -> 2x2 submatrix
          final slice = matrix.slice([
            Slice(start: 1, stop: 3),
            Slice(start: 1, stop: 3),
          ]);
          expect(slice.shape, equals([2, 2]));

          final sendableSlice = slice.toSendableBorrow();

          await Isolate.run(() {
            return NDArray.scope(() {
              setNumThreads(1);
              // Construct non-owning view over the borrowed slice
              final view = sendableSlice.materializeView();
              // Fill the sub-slice with 42.0..45.0
              view.setCell([0, 0], 42.0);
              view.setCell([0, 1], 43.0);
              view.setCell([1, 0], 44.0);
              view.setCell([1, 1], 45.0);
            });
          });

          // Verify that ONLY the interior slice was modified in matrix
          expect(
            matrix.toList(),
            equals([
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              42.0,
              43.0,
              0.0,
              0.0,
              44.0,
              45.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
            ]),
          );
        });
      });

      test('parallel workers mutate non-overlapping slices in-place', () async {
        await NDArray.scope(() async {
          final buffer = NDArray<Int32>.zeros([100], DType.int32);

          // Split into two halves
          final leftSlice = buffer.slice([Slice(start: 0, stop: 50)]);
          final rightSlice = buffer.slice([Slice(start: 50, stop: 100)]);

          final sendableLeft = leftSlice.toSendableBorrow();
          final sendableRight = rightSlice.toSendableBorrow();

          // Concurrently execute two isolate workers mutating their respective slices
          await Future.wait([
            Isolate.run(() {
              NDArray.scope(() {
                final view = sendableLeft.materializeView();
                for (var i = 0; i < view.shape[0]; i++) {
                  view[i] = 1;
                }
              });
            }),
            Isolate.run(() {
              NDArray.scope(() {
                final view = sendableRight.materializeView();
                for (var i = 0; i < view.shape[0]; i++) {
                  view[i] = 2;
                }
              });
            }),
          ]);

          // Verify left half is 1 and right half is 2
          final list = buffer.toList();
          expect(list.sublist(0, 50), equals(List.filled(50, 1)));
          expect(list.sublist(50, 100), equals(List.filled(50, 2)));
        });
      });

      test(
        'materialize() on borrowed array delegates to materializeView',
        () async {
          await NDArray.scope(() async {
            final array = NDArray<Float64>.ones([4], DType.float64);
            final sendable = array.toSendableBorrow();

            await Isolate.run(() {
              return NDArray.scope(() {
                final view = sendable.materialize();
                view[0] = 99.0;
              });
            });

            expect(array.getCell([0]), equals(99.0));
          });
        },
      );

      test(
        'worker disposing materializedView does not free parent memory',
        () async {
          late NDArray<Float64> array;
          NDArray.scope(() {
            array = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            final sendable = array.toSendableBorrow();

            final view = sendable.materializeView();
            // Dispose the view explicitly
            view.dispose();
            expect(view.isDisposed, isTrue);

            // Parent array must still be valid and uncorrupted
            expect(array.isDisposed, isFalse);
            expect(array.toList(), equals([1.0, 2.0, 3.0]));
          });

          // Parent array is properly disposed when its scope ends
          expect(array.isDisposed, isTrue);
        },
      );

      test('throws StateError when attempting to borrow disposed array', () {
        final arr = NDArray<Float64>.ones([5], DType.float64);
        arr.dispose();
        expect(() => arr.toSendableBorrow(), throwsStateError);
        expect(() => SendableNDArray.unsafeBorrow(arr), throwsStateError);
      });

      test(
        'borrows negative-stride (flipped) view accurately across Isolate.run',
        () async {
          await NDArray.scope(() async {
            final arr = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final flipped = NDArray<Float64>.view(
              arr,
              shape: [4],
              strides: [-1],
              offsetElements: 3,
            );
            final sendable = flipped.toSendableBorrow();

            final valuesReadInIsolate = await Isolate.run(() {
              return NDArray.scope(() {
                final view = sendable.materializeView();
                final read = [
                  view.getCell([0]),
                  view.getCell([1]),
                  view.getCell([2]),
                  view.getCell([3]),
                ];
                view.setCell([0], 400.0);
                view.setCell([3], 100.0);
                return read;
              });
            });

            expect(valuesReadInIsolate, equals([40.0, 30.0, 20.0, 10.0]));
            expect(arr.toList(), equals([100.0, 20.0, 30.0, 400.0]));
          });
        },
      );
    });
  });
}
