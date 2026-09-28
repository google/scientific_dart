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

import "dart:ffi" as ffi;
import "package:ffi/ffi.dart";
import "package:ndarray/ndarray.dart";
import "package:test/test.dart";

void main() {
  group("Core NDArray, Indexing & Slicing Workstream 2 Fixes", () {
    group("1. 0-sized dimension slicing", () {
      test(
        "slicing 0-length dimension in 2D array does not crash clamp",
        () => NDArray.scope(() {
          final a = NDArray.zeros([0, 5], DType.float64);
          expect(a.shape, [0, 5]);
          expect(a.size, 0);

          // Basic slicing
          final s1 = a.slice([Slice.all(), Slice.all()]);
          expect(s1.shape, [0, 5]);

          final s2 = a.slice([
            Slice(start: 0, stop: 0),
            Slice(start: 1, stop: 3),
          ]);
          expect(s2.shape, [0, 2]);

          // Advanced indexing along 2nd dimension with 0-sized 1st dimension
          final s3 = a.slice([
            Slice.all(),
            Indices([0, 2, 4]),
          ]);
          expect(s3.shape, [0, 3]);

          // Negative step slicing on 0-sized dimension
          final s4 = a.slice([Slice(step: -1), Slice.all()]);
          expect(s4.shape, [0, 5]);
        }),
      );

      test(
        "slicing 3D array with intermediate 0-sized dimension",
        () => NDArray.scope(() {
          final a = NDArray.zeros([3, 0, 4], DType.int32);
          expect(a.shape, [3, 0, 4]);

          final s1 = a.slice([
            Slice(start: 0, stop: 2),
            Slice.all(),
            Indices([1, 3]),
          ]);
          expect(s1.shape, [2, 0, 2]);

          final s2 = a.take([0, 1], axis: 0);
          expect(s2.shape, [2, 0, 4]);
        }),
      );
    });

    group("2. NDArray.fromList float cast with integers", () {
      test(
        "Float64 array creation with List<int> literals",
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.float64);
          expect(a.dtype, DType.float64);
          expect(a.shape, [2, 2]);
          expect(a.toList(), [1.0, 2.0, 3.0, 4.0]);
          expect(a.getCell([0, 0]), 1.0);
          expect(a.getCell([1, 1]), 4.0);
        }),
      );

      test(
        "Float32 array creation with List<int> literals",
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [10, 20, 30, 40, 50, 60],
            [2, 3],
            DType.float32,
          );
          expect(a.dtype, DType.float32);
          expect(a.shape, [2, 3]);
          expect(a.toList(), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        }),
      );

      test(
        "Float64 array creation with mixed num list",
        () => NDArray.scope(() {
          final dynamic mixedList = [1, 2.5, 3, 4.75];
          final a = NDArray.fromList(mixedList as List, [4], DType.float64);
          expect(a.toList(), [1.0, 2.5, 3.0, 4.75]);
        }),
      );

      test(
        "Float32 array creation with mixed num list",
        () => NDArray.scope(() {
          final dynamic mixedList = [1, 2.5, 3, 4.75];
          final a = NDArray.fromList(mixedList as List, [4], DType.float32);
          expect(a.toList(), [1.0, 2.5, 3.0, 4.75]);
        }),
      );
    });

    group("3. Negative dimension validation in array factories", () {
      test("NDArray.create throws ArgumentError for negative dimensions", () {
        expect(() => NDArray.create([-1], DType.float64), throwsArgumentError);
        expect(() => NDArray.create([2, -3], DType.int32), throwsArgumentError);
        expect(
          () => NDArray.create([-2, -3], DType.uint8),
          throwsArgumentError,
        );
      });

      test(
        "NDArray.zeros and NDArray.ones throw ArgumentError for negative dimensions",
        () {
          expect(
            () => NDArray.zeros([-2, 2], DType.float64),
            throwsArgumentError,
          );
          expect(
            () => NDArray.ones([2, -1], DType.float32),
            throwsArgumentError,
          );
        },
      );

      test(
        "NDArray.fromPointer throws ArgumentError for negative dimensions",
        () {
          final ptr = calloc<ffi.Double>(10);
          try {
            expect(
              () => NDArray.fromPointer(ptr.cast(), [-2, 5], DType.float64),
              throwsArgumentError,
            );
          } finally {
            calloc.free(ptr);
          }
        },
      );

      test(
        "NDArray.view throws ArgumentError for negative dimensions",
        () => NDArray.scope(() {
          final parent = NDArray.zeros([4, 4], DType.float64);
          expect(
            () => NDArray.view(parent, shape: [-2, 2], strides: [2, 1]),
            throwsArgumentError,
          );
        }),
      );
    });

    group("4. NDArray.take negative axis normalization", () {
      test(
        "take with negative axis on 2D array",
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [3, 4],
            DType.int32,
          );

          // axis: -1 should be normalized to axis 1 (columns)
          final cols = a.take([0, 2], axis: -1);
          expect(cols.shape, [3, 2]);
          expect(cols.toList(), [1, 3, 5, 7, 9, 11]);

          // axis: -2 should be normalized to axis 0 (rows)
          final rows = a.take([0, 2], axis: -2);
          expect(rows.shape, [2, 4]);
          expect(rows.toList(), [1, 2, 3, 4, 9, 10, 11, 12]);
        }),
      );

      test(
        "take with negative axis on 3D array",
        () => NDArray.scope(() {
          final a = NDArray.fromList(List.generate(24, (i) => i), [
            2,
            3,
            4,
          ], DType.int32);

          final sub2 = a.take([1, 3], axis: -1); // axis 2
          expect(sub2.shape, [2, 3, 2]);

          final sub1 = a.take([0, 2], axis: -2); // axis 1
          expect(sub1.shape, [2, 2, 4]);

          final sub0 = a.take([1], axis: -3); // axis 0
          expect(sub0.shape, [1, 3, 4]);
        }),
      );

      test(
        "take with out-of-bounds negative axis throws RangeError",
        () => NDArray.scope(() {
          final a = NDArray.zeros([3, 4], DType.float64);
          expect(() => a.take([0], axis: -3), throwsRangeError);
          expect(() => a.take([0], axis: 2), throwsRangeError);
        }),
      );
    });

    group("5. NDEnumerate on sliced views with offsetElements", () {
      test(
        "NDEnumerate on positive sliced view",
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [10, 20, 30, 40, 50, 60, 70, 80, 90],
            [3, 3],
            DType.int32,
          );

          // Sub-matrix [1:, 1:] -> shape [2, 2], values [[50, 60], [80, 90]]
          final view = a.slice([Slice(start: 1), Slice(start: 1)]);
          expect(view.shape, [2, 2]);

          final en = NDEnumerate(view);
          final coords = <List<int>>[];
          final values = <int>[];
          while (en.moveNext()) {
            coords.add(List<int>.from(en.coords));
            values.add(en.value);
          }

          expect(coords, [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
          ]);
          expect(values, [50, 60, 80, 90]);

          // ndenumerate helper function
          final entries = ndenumerate(view).toList();
          expect(entries.map((e) => e.$2).toList(), [50, 60, 80, 90]);
        }),
      );

      test(
        "NDEnumerate on reversed strided sliced view",
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4, 5], [5], DType.int32);
          // Reversed view: [5, 4, 3, 2, 1]
          final rev = a.slice([Slice(step: -1)]);
          expect(rev.shape, [5]);

          final en = NDEnumerate(rev);
          final values = <int>[];
          while (en.moveNext()) {
            values.add(en.value);
          }
          expect(values, [5, 4, 3, 2, 1]);
        }),
      );
    });

    group("6. operator [] coordinate vs row selection ergonomics", () {
      test(
        "1D array fancy list indexing selects elements",
        () => NDArray.scope(() {
          final a = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);

          // List with multiple indices on 1D array extracts elements via take
          final res = a[[1, 3]] as NDArray<AnySpec>;
          expect(res.shape, [2]);
          expect(res.toList(), [20, 40]);

          final res3 = a[[0, 2, 4]] as NDArray<AnySpec>;
          expect(res3.shape, [3]);
          expect(res3.toList(), [10, 30, 50]);

          // Single coordinate lookup
          expect(a[[2]], 30);
        }),
      );

      test(
        "2D array row selection vs coordinate lookup",
        () => NDArray.scope(() {
          final a = NDArray.fromList(List.generate(12, (i) => i), [
            3,
            4,
          ], DType.int32);

          // Coordinate lookup matching rank 2
          expect(a[[0, 1]], 1);
          expect(a[[2, 3]], 11);

          // Coordinate length mismatch throws ArgumentError
          expect(() => a[[1]], throwsArgumentError);

          // Row selection with integer: a[1] extracts row 1 as 1D array
          final row1 = a[1] as NDArray<AnySpec>;
          expect(row1.shape, [4]);
          expect(row1.toList(), [4, 5, 6, 7]);

          // Row stack with nested list: a[[ [0, 2] ]] extracts rows 0 and 2
          final rows02 =
              a[[
                    [0, 2],
                  ]]
                  as NDArray<AnySpec>;
          expect(rows02.shape, [2, 4]);
          expect(rows02.toList(), [0, 1, 2, 3, 8, 9, 10, 11]);
        }),
      );

      test(
        "3D array indexing ergonomics",
        () => NDArray.scope(() {
          final a = NDArray.fromList(List.generate(24, (i) => i), [
            2,
            3,
            4,
          ], DType.int32);

          // Full coordinate matching rank 3
          expect(a[[0, 1, 2]], 6);

          // Subarray selection via slice / integer index
          final sub2D = a[0] as NDArray<AnySpec>;
          expect(sub2D.shape, [3, 4]);
        }),
      );

      test(
        "operator []= polymorphic mutation with ergonomic selectors",
        () => NDArray.scope(() {
          final a = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);
          a[[1, 3]] = NDArray.fromList([99, 88], [2], DType.int32);
          expect(a.toList(), [10, 99, 30, 88, 50]);

          final b = NDArray.fromList(List.generate(12, (i) => i), [
            3,
            4,
          ], DType.int32);
          // Coordinate assignment
          b[[0, 1]] = 100;
          expect(b.getCell([0, 1]), 100);

          // Row slice assignment via a[1]
          b[1] = NDArray.fromList([40, 50, 60, 70], [4], DType.int32);
          expect(b[1].toList(), [40, 50, 60, 70]);
        }),
      );
    });

    group("Stream 1 Core and Native fixes", () {
      test(
        "unique with int16 dtype",
        () => NDArray.scope(() {
          final a = NDArray.fromList([3, 1, 2, 1, 3, 4], [6], DType.int16);
          final u = unique(a);
          expect(u.dtype, DType.int16);
          expect(u.toList(), [1, 2, 3, 4]);

          final multi = NDArray.fromList(
            [5, 5, 2, 2, 9, 1],
            [2, 3],
            DType.int16,
          );
          final uMulti = unique(multi);
          expect(uMulti.dtype, DType.int16);
          expect(uMulti.toList(), [1, 2, 5, 9]);
        }),
      );

      test(
        "contiguity check ignores strides when shape[i] <= 1",
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4, 5], [1, 5], DType.float64);
          expect(a.isContiguous, isTrue);

          final b = NDArray.fromList([1, 2, 3, 4, 5], [5, 1], DType.float64);
          expect(b.isContiguous, isTrue);
        }),
      );

      test("ScratchArena.allocate rejects negative bytes", () {
        expect(() => ScratchArena.allocate(-1), throwsArgumentError);
        expect(() => ScratchArena.allocate(-100), throwsArgumentError);
      });

      test(
        "ScratchArena.copyBools handles BoolList and standard List<bool>",
        () {
          final bools = [true, false, true, true, false];
          final ptr1 = ScratchArena.copyBools(bools);
          final list1 = ptr1.asTypedList(bools.length);
          expect(list1, [1, 0, 1, 1, 0]);

          final arr = NDArray.fromList(bools, [5], DType.boolean);
          final ptr2 = ScratchArena.copyBools(arr.data);
          final list2 = ptr2.asTypedList(bools.length);
          expect(list2, [1, 0, 1, 1, 0]);
        },
      );

      test(
        "cumsum into non-contiguous strided result",
        () => NDArray.scope(() {
          final src = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int16);
          final dest = NDArray.zeros([4, 3], DType.int64);
          final stridedDest = dest.slice([
            Slice(start: 0, stop: 4, step: 2),
            Slice.all(),
          ]);
          expect(stridedDest.isContiguous, isFalse);

          cumsum(src, axis: 1, out: stridedDest);
          expect(stridedDest.toList(), [1, 3, 6, 4, 9, 15]);

          final dest16 = NDArray.zeros([4, 3], DType.int16);
          final stridedDest16 = dest16.slice([
            Slice(start: 0, stop: 4, step: 2),
            Slice.all(),
          ]);
          cumsumAs(src, DType.int16, axis: 1, out: stridedDest16);
          expect(stridedDest16.toList(), [1, 3, 6, 4, 9, 15]);
        }),
      );
    });
  });
}
