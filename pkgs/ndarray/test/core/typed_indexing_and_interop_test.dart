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

import 'dart:typed_data';

import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  tearDown(NDArray.checkNoLeaks);

  group('Typed NDArray operator [] and operator []=', () {
    test('operator [] always returns NDArray<T> across all selector forms', () {
      NDArray.scope(() {
        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );

        // Single int selects row view
        final row1 = mat[1];
        expect(row1, isA<NDArray<Float64>>());
        expect(row1.shape, [3]);
        expect(row1.toList(), [4.0, 5.0, 6.0]);

        // Full coordinates return a 0-D view sharing memory with parent
        final cell = mat[[0, 2]];
        expect(cell, isA<NDArray<Float64>>());
        expect(cell.shape, isEmpty);
        expect(cell.scalar, 3.0);
        cell.fill(33.0);
        expect(mat.getCell([0, 2]), 33.0);

        // Partial per-axis list selects row view (uniform with single int)
        final row0 = mat[[0]];
        expect(row0, isA<NDArray<Float64>>());
        expect(row0.shape, [3]);
        expect(row0.toList(), [1.0, 2.0, 33.0]);

        // Selector objects
        expect(mat[const Index(0)].shape, [3]);
        expect(mat[const Slice.all()].shape, [2, 3]);
        expect(mat[const Slice(start: 0, stop: 1)].shape, [1, 3]);
        expect(mat[Indices([1, 0])].toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 33.0]);

        // Boolean NDArray & Mask(NDArray<Boolean>)
        final mask = mat > 3.0;
        expect(mat[mask].toList(), [33.0, 4.0, 5.0, 6.0]);
        expect(mat[Mask(mask)].toList(), [33.0, 4.0, 5.0, 6.0]);
        mat[Mask(mask)] = 7.0;
        expect(mat.toList(), [1.0, 2.0, 7.0, 7.0, 7.0, 7.0]);
        mat[Mask(mask)] = NDArray.fromList(
          [33.0, 4.0, 5.0, 6.0],
          [4],
          DType.float64,
        );

        // Axis mask in list
        final rowMask = NDArray.fromList([false, true], [2], DType.boolean);
        expect(mat[[Mask(rowMask), const Slice.all()]].toList(), [
          4.0,
          5.0,
          6.0,
        ]);

        // Integer NDArray fancy indexing
        final idx = NDArray.fromList([1, 0], [2], DType.int32);
        expect(mat[idx].shape, [2, 3]);
        expect(mat[idx].toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 33.0]);

        // Too many per-axis selectors throws ArgumentError
        expect(() => mat[[0, 1, 2]], throwsArgumentError);

        // 1-D flat list > rank 1 throws ArgumentError; use Indices or nested list
        final vec = NDArray.fromList([10, 20, 30, 40], [4], DType.int32);
        expect(() => vec[[1, 3]], throwsArgumentError);
        expect(vec[Indices([1, 3])].toList(), [20, 40]);
        expect(
          vec[[
                [1, 3],
              ]]
              .toList(),
          [20, 40],
        );
        expect(vec[[2]].scalar, 30);
      });
    });

    test('Mask validates disposed mask and mismatched shape', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final dispMask = NDArray.fromList(
          [true, false, true],
          [3],
          DType.boolean,
        )..dispose();
        expect(() => a[Mask(dispMask)], throwsStateError);
        expect(() => a[Mask(dispMask)] = 0.0, throwsStateError);
        expect(() => a.slice([Mask(dispMask)]), throwsStateError);

        final wrongMask = NDArray.fromList([true, false], [2], DType.boolean);
        expect(() => a[Mask(wrongMask)], throwsArgumentError);
        expect(() => a[Mask(wrongMask)] = 0.0, throwsArgumentError);
      });
    });

    test(
      'typed setByMask, setIndices, setByMaskScalar, setIndicesScalar & dynamic dispatch',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final mask = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          a.setByMaskScalar(mask, -1.0);
          expect(a.toList(), [-1.0, 2.0, -1.0, 4.0]);

          final vals = NDArray.fromList([10.0, 30.0], [2], DType.float64);
          a.setByMask(mask, vals);
          expect(a.toList(), [10.0, 2.0, 30.0, 4.0]);

          final idx = NDArray.fromList([1, 3], [2], DType.int32);
          a.setIndicesScalar(idx, 99.0);
          expect(a.toList(), [10.0, 99.0, 30.0, 99.0]);

          a.setIndices(idx, NDArray.fromList([20.0, 40.0], [2], DType.float64));
          expect(a.toList(), [10.0, 20.0, 30.0, 40.0]);

          // Dynamic dispatch via noSuchMethod
          final dynamic dynA = a;
          dynA.setByMaskScalar(mask, 5.0);
          expect(a.toList(), [5.0, 20.0, 5.0, 40.0]);
          dynA.setIndicesScalar(idx, 7.0);
          expect(a.toList(), [5.0, 7.0, 5.0, 7.0]);

          // Complex scalar mask/index mutation
          final c = NDArray.fromList(
            [const Complex(1, 2), const Complex(3, 4)],
            [2],
            DType.complex128,
          );
          final cMask = NDArray.fromList([false, true], [2], DType.boolean);
          c.setByMaskScalar(cMask, const Complex(9, -1));
          expect(c.toList(), [const Complex(1, 2), const Complex(9, -1)]);

          // Untyped NDArray<DTypeTag> fallback extension
          final NDArray<DTypeTag> erased = a.asAnySpec;
          erased.setByMaskScalar(mask, 42.0);
          expect(a.toList(), [42.0, 7.0, 42.0, 7.0]);
        });
      },
    );

    test('operator []= handles coordinates, slices, read-only, and errors', () {
      NDArray.scope(() {
        final m = NDArray.zeros([2, 3], DType.int32);
        m[[0, 1]] = 7;
        expect(m.getCell([0, 1]), 7);

        // Assign 0-D NDArray to coordinate; rank-1 [1] cannot broadcast to 0-D []
        m[[1, 2]] = NDArray.scalar(11, dtype: DType.int32);
        expect(m.getCell([1, 2]), 11);
        expect(
          () => m[[1, 2]] = NDArray.fromList([11], [1], DType.int32),
          throwsArgumentError,
        );

        // Single-element per-axis list assigns row
        m[[1]] = 5;
        expect(m[1].toList(), [5, 5, 5]);

        // Too many selectors throws ArgumentError
        expect(() => m[[0, 1, 2]] = 9, throwsArgumentError);

        // Read-only throws StateError
        m.isWriteable = false;
        expect(() => m[[0, 0]] = 1, throwsStateError);
        expect(() => m[0] = 1, throwsStateError);
      });
    });
  });

  group('NDArray.fromTypedData & NDArray.fromBytes', () {
    test(
      'fromTypedData round-trips all 15 dtypes and copies independently',
      () {
        NDArray.scope(() {
          final f64Src = Float64List.fromList([1.5, -2.5, 3.25, 4.75]);
          final f64 = NDArray.fromTypedData(f64Src, [2, 2], DType.float64);
          f64Src[0] = 999.0; // source mutation must not affect NDArray
          expect(f64.toList(), [1.5, -2.5, 3.25, 4.75]);
          expect(f64.toTypedData(), isA<Float64List>());
          expect(f64.toTypedData(), [1.5, -2.5, 3.25, 4.75]);

          final f32 = NDArray.fromTypedData(Float32List.fromList([1.0, 2.5]), [
            2,
          ], DType.float32);
          expect(f32.toTypedData(), isA<Float32List>());
          expect(f32.toList(), [1.0, 2.5]);

          // Float16 raw bits: 0x3C00 = 1.0, 0x4000 = 2.0
          final f16 = NDArray.fromTypedData(
            Uint16List.fromList([0x3C00, 0x4000]),
            [2],
            DType.float16,
          );
          expect(f16.toList(), [1.0, 2.0]);
          expect(f16.toTypedData(), isA<Uint16List>());
          expect(f16.toTypedData(), [0x3C00, 0x4000]);

          // BFloat16 raw bits: 0x3F80 = 1.0, 0x4000 = 2.0
          final bf16 = NDArray.fromTypedData(
            Uint16List.fromList([0x3F80, 0x4000]),
            [2],
            DType.bfloat16,
          );
          expect(bf16.toList(), [1.0, 2.0]);
          expect(bf16.toTypedData(), isA<Uint16List>());
          expect(bf16.toTypedData(), [0x3F80, 0x4000]);

          final i64 = NDArray.fromTypedData(Int64List.fromList([-100, 200]), [
            2,
          ], DType.int64);
          expect(i64.toTypedData(), isA<Int64List>());
          expect(i64.toList(), [-100, 200]);

          final i32 = NDArray.fromTypedData(Int32List.fromList([-10, 20]), [
            2,
          ], DType.int32);
          expect(i32.toTypedData(), isA<Int32List>());
          expect(i32.toList(), [-10, 20]);

          final i16 = NDArray.fromTypedData(Int16List.fromList([-300, 300]), [
            2,
          ], DType.int16);
          expect(i16.toTypedData(), isA<Int16List>());
          expect(i16.toList(), [-300, 300]);

          final i8 = NDArray.fromTypedData(Int8List.fromList([-128, 127]), [
            2,
          ], DType.int8);
          expect(i8.toTypedData(), isA<Int8List>());
          expect(i8.toList(), [-128, 127]);

          final u64 = NDArray.fromTypedData(Uint64List.fromList([0, 42]), [
            2,
          ], DType.uint64);
          expect(u64.toTypedData(), isA<Uint64List>());
          expect(u64.toList(), [0, 42]);

          final u32 = NDArray.fromTypedData(
            Uint32List.fromList([1, 4000000000]),
            [2],
            DType.uint32,
          );
          expect(u32.toTypedData(), isA<Uint32List>());
          expect(u32.toList(), [1, 4000000000]);

          final u16 = NDArray.fromTypedData(Uint16List.fromList([0, 65535]), [
            2,
          ], DType.uint16);
          expect(u16.toTypedData(), isA<Uint16List>());
          expect(u16.toList(), [0, 65535]);

          final u8 = NDArray.fromTypedData(Uint8List.fromList([0, 255]), [
            2,
          ], DType.uint8);
          expect(u8.toTypedData(), isA<Uint8List>());
          expect(u8.toList(), [0, 255]);

          // Uint8ClampedList also accepted for DType.uint8
          final u8Clamped = NDArray.fromTypedData(
            Uint8ClampedList.fromList([10, 250]),
            [2],
            DType.uint8,
          );
          expect(u8Clamped.toList(), [10, 250]);

          // Interleaved complex128 / complex64
          final c128 = NDArray.fromTypedData(
            Float64List.fromList([1.0, 2.0, -3.0, 4.0]),
            [2],
            DType.complex128,
          );
          expect(c128.toList(), [
            const Complex(1.0, 2.0),
            const Complex(-3.0, 4.0),
          ]);
          expect(c128.toTypedData(), isA<Float64List>());
          expect(c128.toTypedData(), [1.0, 2.0, -3.0, 4.0]);

          final c64 = NDArray.fromTypedData(
            Float32List.fromList([1.5, -0.5, 2.0, 3.0]),
            [2],
            DType.complex64,
          );
          expect(c64.toList(), [
            const Complex(1.5, -0.5),
            const Complex(2.0, 3.0),
          ]);
          expect(c64.toTypedData(), isA<Float32List>());
          expect(c64.toTypedData(), [1.5, -0.5, 2.0, 3.0]);

          // Boolean normalizes non-zero bytes to 1
          final b = NDArray.fromTypedData(Uint8List.fromList([0, 1, 2, 255]), [
            4,
          ], DType.boolean);
          expect(b.toList(), [false, true, true, true]);
          expect(b.toTypedData(), isA<Uint8List>());
          expect(b.toTypedData(), [0, 1, 1, 1]);
          expect(b.toBytes(), [0, 1, 1, 1]);
        });
      },
    );

    test('fromTypedData respects non-zero offsetInBytes on sublist views', () {
      NDArray.scope(() {
        final big = Float64List.fromList([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        final sub = Float64List.sublistView(big, 2, 5);
        final arr = NDArray.fromTypedData(sub, [3], DType.float64);
        expect(arr.toList(), [30.0, 40.0, 50.0]);

        final bigBool = Uint8List.fromList([1, 0, 7, 0, 99]);
        final subBool = Uint8List.sublistView(bigBool, 2, 5);
        final boolArr = NDArray.fromTypedData(subBool, [3], DType.boolean);
        expect(boolArr.toList(), [true, false, true]);
        expect(boolArr.toBytes(), [1, 0, 1]);
      });
    });

    test('fromTypedData handles empty shapes and rejects invalid inputs', () {
      NDArray.scope(() {
        final empty = NDArray.fromTypedData(Float64List(0), [
          2,
          0,
          3,
        ], DType.float64);
        expect(empty.shape, [2, 0, 3]);
        expect(empty.size, 0);

        // Wrong TypedData subtype
        expect(
          () => NDArray.fromTypedData(Float32List(4), [4], DType.float64),
          throwsArgumentError,
        );
        expect(
          () => NDArray.fromTypedData(Int32List(4), [4], DType.int64),
          throwsArgumentError,
        );
        // Wrong length
        expect(
          () => NDArray.fromTypedData(Float64List(3), [2, 2], DType.float64),
          throwsArgumentError,
        );
        // Odd length for complex
        expect(
          () => NDArray.fromTypedData(Float64List(3), [1], DType.complex128),
          throwsArgumentError,
        );
        expect(
          () => NDArray.fromTypedData(Float32List(5), [2], DType.complex64),
          throwsArgumentError,
        );
        // Negative shape
        expect(
          () => NDArray.fromTypedData(Float64List(0), [-1], DType.float64),
          throwsArgumentError,
        );
      });
    });

    test('fromBytes and toBytes round-trip, sublist offset, and errors', () {
      NDArray.scope(() {
        final src = NDArray.fromList(
          [1.25, -3.5, 0.0, 42.0],
          [2, 2],
          DType.float64,
        );
        final bytes = src.toBytes();
        expect(bytes.length, 4 * 8);

        final roundTrip = NDArray.fromBytes(bytes, [2, 2], DType.float64);
        expect(roundTrip.toList(), [1.25, -3.5, 0.0, 42.0]);

        // Sublist view with non-zero offsetInBytes
        final padded = Uint8List(bytes.length + 16);
        padded.setAll(8, bytes);
        final subBytes = Uint8List.sublistView(padded, 8, 8 + bytes.length);
        final fromSub = NDArray.fromBytes(subBytes, [2, 2], DType.float64);
        expect(fromSub.toList(), [1.25, -3.5, 0.0, 42.0]);

        // Boolean normalization
        final b = NDArray.fromBytes(Uint8List.fromList([0, 42, 0, 255]), [
          4,
        ], DType.boolean);
        expect(b.toList(), [false, true, false, true]);
        expect(b.toBytes(), [0, 1, 0, 1]);

        // Empty
        final empty = NDArray.fromBytes(Uint8List(0), [0, 4], DType.int32);
        expect(empty.shape, [0, 4]);
        expect(empty.toBytes(), isEmpty);

        // Wrong byte length
        expect(
          () => NDArray.fromBytes(Uint8List(7), [1], DType.float64),
          throwsArgumentError,
        );
        // Negative shape
        expect(
          () => NDArray.fromBytes(Uint8List(0), [-2], DType.uint8),
          throwsArgumentError,
        );
      });
    });
  });

  group('Zero-copy views (asTypedDataView, withTypedData, asBytesView)', () {
    test('zero-copy live mutation and sliced contiguous view offsets', () {
      NDArray.scope(() {
        final base = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
          [2, 3],
          DType.float64,
        );

        // Contiguous slice starting at offset 3 (row 1)
        final row1 = base[1];
        expect(row1.isContiguous, isTrue);

        final view = row1.asTypedDataView();
        expect(view, isA<Float64List>());
        expect(view.length, 3);
        expect(view, [40.0, 50.0, 60.0]);

        // Mutating view mutates underlying base array at row 1 (not row 0!)
        view[1] = 555.0;
        expect(base.getCell([0, 1]), 20.0);
        expect(base.getCell([1, 1]), 555.0);

        // withTypedData receives scoped view and returns callback result
        final total = row1.withTypedData((v) {
          v[2] = 666.0;
          return v[0] + v[1] + v[2];
        });
        expect(total, 40.0 + 555.0 + 666.0);
        expect(base.getCell([1, 2]), 666.0);

        // asBytesView respects slice offset
        final rowBytes = row1.asBytesView();
        expect(rowBytes.length, 3 * 8);
        expect(rowBytes, equals(row1.toBytes()));

        // Dynamic dispatch via noSuchMethod
        final dynamic dynRow = row1;
        expect(dynRow.toTypedData(), [40.0, 555.0, 666.0]);
        expect(dynRow.asTypedDataView(), [40.0, 555.0, 666.0]);
        expect(dynRow.withTypedData((Float64List v) => v.length), 3);

        // Erased NDArray<DTypeTag> extension
        final NDArray<DTypeTag> erased = row1.asAnySpec;
        expect(erased.toTypedData(), isA<Float64List>());
        expect(erased.asTypedDataView(), isA<Float64List>());
        expect(erased.withTypedData((v) => v.lengthInBytes), 24);
      });
    });

    test(
      'asTypedDataView works across all 15 dtypes including read-only enforcement',
      () {
        NDArray.scope(() {
          void checkWriteAndReadOnly<T extends DTypeTag, V extends TypedData>(
            NDArray<T> arr,
            V Function(NDArray<T>) getView,
            void Function(V) mutate,
          ) {
            final v = getView(arr);
            expect(v.lengthInBytes, arr.size * arr.dtype.byteWidth);
            mutate(v);

            arr.isWriteable = false;
            final roView = getView(arr);
            expect(roView.lengthInBytes, arr.size * arr.dtype.byteWidth);
            expect(() => mutate(roView), throwsUnsupportedError);

            final roBytes = arr.asBytesView();
            expect(() => roBytes[0] = 1, throwsUnsupportedError);
          }

          checkWriteAndReadOnly(
            NDArray.fromList([1.0, 2.0], [2], DType.float64),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9.0,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1.0, 2.0], [2], DType.float32),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9.0,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1.0, 2.0], [2], DType.float16),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 0x3C00,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1.0, 2.0], [2], DType.bfloat16),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 0x3F80,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.int64),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.int32),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.int16),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.int8),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.uint64),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.uint32),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.uint16),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([1, 2], [2], DType.uint8),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([const Complex(1, 2)], [1], DType.complex128),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9.0,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([const Complex(1, 2)], [1], DType.complex64),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 9.0,
          );
          checkWriteAndReadOnly(
            NDArray.fromList([true, false], [2], DType.boolean),
            (a) => a.asTypedDataView(),
            (v) => v[0] = 1,
          );
        });
      },
    );

    test(
      'non-contiguous strided views copy on toTypedData/toBytes and throw on views',
      () {
        NDArray.scope(() {
          final m = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final transposed = m.transpose();
          expect(transposed.isContiguous, isFalse);

          // Copying methods materialize logical C-order
          expect(transposed.toTypedData(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
          final reconstructed = NDArray.fromBytes(transposed.toBytes(), [
            3,
            2,
          ], DType.float64);
          expect(reconstructed.toList(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

          // Zero-copy views reject non-contiguous layouts
          expect(() => transposed.asTypedDataView(), throwsStateError);
          expect(() => transposed.asBytesView(), throwsStateError);
          expect(
            () => transposed.withTypedData((v) => v.length),
            throwsStateError,
          );

          // Broadcast view is also non-contiguous
          final bcast = broadcastTo(
            NDArray.fromList([7.0], [1], DType.float64),
            [3],
          );
          expect(bcast.isContiguous, isFalse);
          expect(bcast.toTypedData(), [7.0, 7.0, 7.0]);
          expect(() => bcast.asTypedDataView(), throwsStateError);
          expect(() => bcast.asBytesView(), throwsStateError);
        });
      },
    );

    test('empty and disposed arrays behave consistently', () {
      NDArray.scope(() {
        final empty = NDArray.zeros([0], DType.float64);
        expect(empty.toTypedData(), isEmpty);
        expect(empty.asTypedDataView(), isEmpty);
        expect(empty.toBytes(), isEmpty);
        expect(empty.asBytesView(), isEmpty);
        expect(empty.withTypedData((v) => v.length), 0);

        empty.isWriteable = false;
        expect(empty.asTypedDataView(), isEmpty);
        expect(empty.asBytesView(), isEmpty);

        empty.dispose();
        expect(() => empty.toTypedData(), throwsStateError);
        expect(() => empty.asTypedDataView(), throwsStateError);
        expect(() => empty.toBytes(), throwsStateError);
        expect(() => empty.asBytesView(), throwsStateError);
        expect(() => empty.withTypedData((v) => v.length), throwsStateError);
      });
    });

    test(
      'Float16List and BFloat16List data views support read/write cleanly',
      () {
        NDArray.scope(() {
          final f16 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float16);
          final f16Data = f16.data;
          expect(f16Data, isA<Float16List>());
          expect(f16Data[0], 1.0);
          f16Data[1] = 4.0;
          expect(f16.getCell([1]), 4.0);

          final bf16 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.bfloat16);
          final bf16Data = bf16.data;
          expect(bf16Data, isA<BFloat16List>());
          expect(bf16Data[0], 1.0);
          bf16Data[2] = 8.0;
          expect(bf16.getCell([2]), 8.0);
        });
      },
    );
  });

  group('Fast-path fromList and toList', () {
    test(
      'fromList delegates on matching TypedData and validates general lists without leaking',
      () {
        NDArray.scope(() {
          // Matching TypedData sublist view fast-path
          final big = Float64List.fromList([0.0, 1.5, 2.5, 3.5, 9.0]);
          final sub = Float64List.sublistView(big, 1, 4);
          final fromSub = NDArray.fromList(sub, [3], DType.float64);
          expect(fromSub.toList(), [1.5, 2.5, 3.5]);

          // Non-matching TypedData (Int32List -> Float64) still converts element-wise
          final converted = NDArray.fromList(Int32List.fromList([1, 2, 3]), [
            3,
          ], DType.float64);
          expect(converted.toList(), [1.0, 2.0, 3.0]);

          // Out-of-range integer value throws ArgumentError without leaking
          expect(
            () => NDArray.fromList([0, 256], [2], DType.uint8),
            throwsArgumentError,
          );
          expect(
            () => NDArray.fromList([-1, 0], [2], DType.uint8),
            throwsArgumentError,
          );
          expect(
            () => NDArray.fromList([-129, 0], [2], DType.int8),
            throwsArgumentError,
          );
          // Wrong element type throws TypeError without leaking
          expect(
            () => NDArray.fromList(['bad', 'data'], [2], DType.float64),
            throwsA(isA<TypeError>()),
          );
        });
      },
    );

    test(
      'toList returns growable list for contiguous, sliced, strided, and empty arrays',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final list = a.toList();
          expect(list, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
          list.add(7.0); // must be growable
          expect(list.length, 7);

          // Contiguous sliced view with offset
          final row1List = a[1].toList();
          expect(row1List, [4.0, 5.0, 6.0]);
          row1List.add(7.0);
          expect(row1List.length, 4);

          // Strided view
          final tList = a.transpose().toList();
          expect(tList, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
          tList.add(9.0);
          expect(tList.length, 7);

          // Complex128 contiguous slice with offset
          final c = NDArray.fromList(
            [
              const Complex(1, 2),
              const Complex(3, 4),
              const Complex(5, 6),
              const Complex(7, 8),
            ],
            [2, 2],
            DType.complex128,
          );
          expect(c[1].toList(), [const Complex(5, 6), const Complex(7, 8)]);

          // Empty
          final emptyList = NDArray.zeros([0], DType.int32).toList();
          expect(emptyList, isEmpty);
          emptyList.add(1);
          expect(emptyList, [1]);
        });
      },
    );
  });
}
