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
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray.dart'
    show
        BoolList,
        ComplexList,
        NDArrayBaseDivide,
        NDArrayBaseElements,
        NDArrayBitwiseSpec,
        NDArrayShiftSpec,
        listEquals,
        uint64Compare;
import 'package:test/test.dart';

void main() {
  group('NDArray Core Branch Coverage', () {
    test('relative span & stride overflow checks', () {
      NDArray.scope(() {
        expect(
          () => NDArray.create(
            [2, 2],
            DType.uint8,
            strides: [0x5000000000000000, 0x5000000000000000],
          ),
          throwsArgumentError,
        );
        expect(
          () =>
              NDArray.create([3], DType.uint8, strides: [-0x5000000000000000]),
          throwsArgumentError,
        );
        expect(
          () => NDArray.create(
            [2, 2],
            DType.uint8,
            strides: [-0x5000000000000000, -0x5000000000000000],
          ),
          throwsArgumentError,
        );
        expect(
          () => NDArray.create(
            [2, 2],
            DType.uint8,
            strides: [0x5000000000000000, -0x5000000000000000],
          ),
          throwsArgumentError,
        );
        expect(
          () => NDArray.create([2, 2], DType.float64, strides: [1]),
          throwsArgumentError,
        );
        expect(() => NDArray.computeCStrides([2, -1]), throwsArgumentError);
      });
    });

    test('disposed array errors, capacity, and isWriteable parent checks', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2], DType.float64);
        a.dispose();
        expect(() => a.getCellRaw(0), throwsStateError);
        expect(() => a.physicalByteCapacity, throwsStateError);
        expect(() => a.asBitwiseDType, throwsStateError);
        expect(() => a.asIntegerDType, throwsStateError);

        final base = NDArray.zeros([1], DType.float64);
        final bcast = broadcastTo(base, [3]);
        final sliceOfBcast = bcast.slice([const Slice(start: 0, stop: 1)]);
        expect(() => sliceOfBcast.isWriteable = true, throwsArgumentError);
      });
    });

    test(
      'factories: full error cleanup, arange overflow, view & fromPointer',
      () {
        NDArray.scope(() {
          expect(
            () => NDArray.full([2], 'bad', dtype: DType.float64),
            throwsA(anything),
          );
          expect(
            () => NDArray.arange(0, 1e20, step: 1e-10, dtype: DType.float64),
            throwsArgumentError,
          );

          final a = NDArray.zeros([4], DType.float64);
          final disp = NDArray.zeros([2], DType.float64)..dispose();
          expect(
            () => NDArray.view(disp, shape: [2], strides: [1]),
            throwsStateError,
          );
          expect(
            () => NDArray.view(a, shape: [2], strides: [1, 1]),
            throwsArgumentError,
          );
          expect(
            () => NDArray.view(
              a,
              shape: [1],
              strides: [1],
              offsetElements: 0x4000000000000000,
            ),
            throwsArgumentError,
          );
          final u8 = NDArray.zeros([4], DType.uint8);
          expect(
            () => NDArray.view(
              u8,
              shape: [2],
              strides: [-0x7ffffffffffffffe],
              offsetElements: -2,
            ),
            throwsRangeError,
          );
          expect(
            () => NDArray.view(
              u8,
              shape: [2],
              strides: [0x7ffffffffffffffe],
              offsetElements: 2,
            ),
            throwsRangeError,
          );

          final ptr = calloc<ffi.Uint8>(64);
          try {
            expect(
              () => NDArray.fromPointer(
                ptr.cast(),
                [2, 2],
                DType.float64,
                strides: [1],
              ),
              throwsArgumentError,
            );
            for (final dt in [
              DType.bfloat16,
              DType.uint64,
              DType.uint32,
              DType.uint16,
              DType.uint8,
              DType.boolean,
            ]) {
              final ext = NDArray.fromPointer(ptr.cast(), [2], dt);
              expect(ext.shape, [2]);
            }
          } finally {
            calloc.free(ptr);
          }
        });
      },
    );

    test('reshape, copy, copyToContiguous, fillUntyped, and scalarRaw', () {
      NDArray.scope(() {
        final a = NDArray.zeros([3], DType.float64);
        expect(() => a.reshape([0x4000000000000000, 4]), throwsArgumentError);
        expect(() => a.reshape([-1, 0]), throwsArgumentError);
        expect(() => a.reshape([-1, 2]), throwsArgumentError);

        final disp = NDArray.zeros([3], DType.float64)..dispose();
        expect(() => a.copy(out: disp), throwsStateError);
        final i64 = NDArray.zeros([3], DType.int64);
        expect(a.copy(out: a), same(a));

        expect(() => disp.copyToContiguous(a), throwsStateError);
        expect(() => a.copyToContiguous(disp), throwsStateError);
        expect(() => a.copyToContiguous(i64), throwsArgumentError);
        final a2 = NDArray.zeros([4], DType.float64);
        expect(() => a.copyToContiguous(a2), throwsArgumentError);
        final empty1 = NDArray.zeros([0], DType.float64);
        final empty2 = NDArray.zeros([0], DType.float64);
        empty1.copyToContiguous(empty2);
        final strided = a2.slice([const Slice(step: 2)]);
        final dst2 = NDArray.zeros([2], DType.float64);
        expect(() => strided.copyToContiguous(dst2), throwsArgumentError);
        expect(() => dst2.copyToContiguous(strided), throwsArgumentError);

        final bf16 = NDArray.zeros([4], DType.bfloat16);
        final bf16Strided = bf16.slice([const Slice(step: 2)]);
        bf16Strided.fill(2.0);
        expect(bf16.toList(), [2.0, 0.0, 2.0, 0.0]);

        expect(() => a.scalar, throwsStateError);

        strided.setCellFlat(1, 9.0);
        expect(a2.toList(), [0.0, 0.0, 9.0, 0.0]);
      });
    });

    test(
      'setByMask, setByMaskScalar, setIndices, setIndicesScalar & coerceScalar',
      () {
        NDArray.scope(() {
          final a = NDArray.zeros([2, 2], DType.float64);
          final m1 = NDArray.fromList([true, false], [2], DType.boolean);
          final mWrong = NDArray.fromList(
            [true, false, true],
            [1, 3],
            DType.boolean,
          );
          final v = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => a.setByMask(m1, v), throwsArgumentError);
          expect(() => a.setByMask(mWrong, v), throwsArgumentError);

          final m2 = NDArray.fromList(
            [true, false, false, true],
            [2, 2],
            DType.boolean,
          );
          final v0d = NDArray.scalar(7.0, dtype: DType.float64);
          a.setByMask(m2, v0d);
          expect(a.toList(), [7.0, 0.0, 0.0, 7.0]);

          final bArr = NDArray.fromList([true, false], [2], DType.boolean);
          final bVal = NDArray.fromList([false], [1], DType.boolean);
          bArr.setByMask(bArr, bVal);
          expect(bArr.toList(), [false, false]);

          final bArr2 = NDArray.fromList([true, false], [2], DType.boolean);
          bArr2.setByMaskScalar(bArr2, false);
          expect(bArr2.toList(), [false, false]);

          a.setByMaskScalar(m2, NDArray.scalar(3.0, dtype: DType.float64));
          expect(a.toList(), [3.0, 0.0, 0.0, 3.0]);
          expect(() => a.setByMaskScalar(m2, v), throwsArgumentError);

          final iArr = NDArray.fromList([1, 0], [2], DType.int64);
          iArr.setIndicesScalar(iArr, 5);
          expect(iArr.toList(), [5, 5]);

          iArr.setIndicesScalar(
            NDArray.fromList([0], [1], DType.int64),
            NDArray.scalar(9, dtype: DType.int64),
          );
          expect(iArr.toList(), [9, 5]);
          expect(
            () => iArr.setIndicesScalar(
              NDArray.fromList([0], [1], DType.int64),
              NDArray.fromList([1, 2], [2], DType.int64),
            ),
            throwsArgumentError,
          );

          final negU64 = NDArray.fromList([-1], [1], DType.uint64);
          expect(() => iArr.setIndicesScalar(negU64, 1), throwsRangeError);
          expect(() => iArr.setIndices(negU64, iArr), throwsRangeError);

          iArr.setIndices(
            NDArray.fromList([-1], [1], DType.int64),
            NDArray.scalar(11, dtype: DType.int64),
          );
          expect(iArr.toList(), [9, 11]);

          final iArr2 = NDArray.fromList([1, 0], [2], DType.int64);
          iArr2.setIndices(iArr2, NDArray.fromList([10, 20], [2], DType.int64));
          expect(iArr2.toList(), [20, 10]);

          // _coerceScalar branches via []= on single coordinate
          final disp0d = NDArray.scalar(1.0, dtype: DType.float64)..dispose();
          expect(() => a[[0, 0]] = disp0d, throwsStateError);
          final u64Scalar = NDArray.scalar(4, dtype: DType.uint64);
          a[[0, 0]] = u64Scalar;
          expect(a.getCell([0, 0]), 4.0);
          a[[0, 0]] = true;
          expect(a.getCell([0, 0]), 1.0);
          a[[0, 0]] = const Complex(5.0, 0.0);
          expect(a.getCell([0, 0]), 5.0);

          iArr[[0]] = 2.5;
          expect(iArr.getCell([0]), 2);
          iArr[[0]] = true;
          expect(iArr.getCell([0]), 1);
          iArr[[0]] = const Complex(6.0, 0.0);
          expect(iArr.getCell([0]), 6);

          final cArr = NDArray.zeros([2], DType.complex128);
          cArr[[0]] = u64Scalar;
          expect(cArr.getCell([0]), const Complex(4.0, 0.0));
          cArr[[0]] = 8;
          expect(cArr.getCell([0]), const Complex(8.0, 0.0));
          cArr[[0]] = true;
          expect(cArr.getCell([0]), const Complex(1.0, 0.0));

          bArr[[0]] = 1;
          expect(bArr.getCell([0]), isTrue);
          bArr[[0]] = const Complex(0.0, 1.0);
          expect(bArr.getCell([0]), isTrue);
          bArr[[0]] = const Complex(0.0, 0.0);
          expect(bArr.getCell([0]), isFalse);
        });
      },
    );

    test('indexing selectors, slice, sliceAssign, operator [] and []=', () {
      NDArray.scope(() {
        final a = NDArray.arange(0, 6, dtype: DType.float64).reshape([2, 3]);
        final stridedIdx = NDArray.fromList(
          [0, 99, 1],
          [3],
          DType.int64,
        ).slice([const Slice(step: 2)]);
        expect((a[stridedIdx] as NDArray).shape, [2, 3]);

        expect((a[[<int>[], const Slice.all()]] as NDArray).shape, [0, 3]);
        expect(
          (a[[
                    [true, false],
                    const Slice.all(),
                  ]]
                  as NDArray)
              .shape,
          [1, 3],
        );
        a[[
              [true, false],
              const Slice.all(),
            ]] =
            99.0;
        expect(a.getCell([0, 0]), 99.0);

        final m1d = NDArray.fromList([false, true], [2], DType.boolean);
        expect((a[[m1d, const Slice.all()]] as NDArray).shape, [1, 3]);
        expect(
          (a[[
                    NDArray.fromList([0], [1], DType.int32),
                    const Slice.all(),
                  ]]
                  as NDArray)
              .shape,
          [1, 3],
        );
        expect(
          () =>
              a[[
                NDArray.fromList([0.0], [1], DType.float64),
              ]],
          throwsArgumentError,
        );
        expect((a[BooleanMask(m1d)] as NDArray).shape, [1, 3]);
        a[BooleanMask(m1d)] = 42.0;
        expect(a.getCell([1, 0]), 42.0);

        final m1dWrong = NDArray.fromList(
          [true, false, true],
          [3],
          DType.boolean,
        );
        expect(() => a[BooleanMask(m1dWrong)] = 1.0, throwsArgumentError);
        expect(
          () => a.slice([Mask(BooleanMask(m1dWrong))]),
          throwsArgumentError,
        );
        expect(
          () => a.slice([const Index(0), const Index(0), const Index(0)]),
          throwsArgumentError,
        );
        expect(() => a.slice([const Index(99)]), throwsRangeError);
        expect(
          () => a.slice([
            Indices([99]),
          ]),
          throwsRangeError,
        );

        final disp = NDArray.zeros([2], DType.float64)..dispose();
        expect(() => disp.sliceAssign([], 1.0), throwsStateError);
        expect(() => a.sliceAssign([const Index(0)], disp), throwsStateError);
        expect(() => a[0] = disp, throwsStateError);

        a[const Slice.all()] = NDArray.scalar(5.0, dtype: DType.float64);
        expect(a.getCell([0, 0]), 5.0);

        a[[
          Indices([0, 1]),
          const Slice(step: -1),
        ]] = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        expect(a.getCell([0, 2]), 1.0);
        a[[
              Indices([0, 1]),
              const Slice.all(),
            ]] =
            10.0;
        expect(a.getCell([0, 0]), 10.0);

        expect((a[m1d] as NDArray).shape, [1, 3]);
        expect(a.applyMask(m1d).shape, [1, 3]);
        expect(() => a.applyMask(m1dWrong), throwsArgumentError);
        expect(
          NDArray.zeros([
            0,
          ], DType.float64).applyMask(NDArray.zeros([0], DType.boolean)).shape,
          [0],
        );

        a[m1d] = 77.0;
        expect(a.getCell([1, 0]), 77.0);
        expect(() => a[m1dWrong] = 1.0, throwsArgumentError);

        a[0] = NDArray.fromList([1.0, 2.0, 3.0], [1, 3], DType.float64);
        expect(a.getCell([0, 1]), 2.0);
        a[0] = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
        expect(a.getCell([0, 1]), 5.0);

        final v1d = NDArray.zeros([4], DType.float64);
        v1d[[0, 2]] = NDArray.fromList([8.0, 9.0], [2], DType.float64);
        expect(v1d.toList(), [8.0, 0.0, 9.0, 0.0]);
        v1d[[1, 3]] = NDArray.fromList([3.0], [1], DType.float64);
        expect(v1d.toList(), [8.0, 3.0, 9.0, 3.0]);

        final fullMask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        v1d[fullMask] = NDArray.fromList([11.0], [1], DType.float64);
        expect(v1d.toList(), [11.0, 3.0, 11.0, 3.0]);

        final u64Idx = NDArray.fromList([0, 1], [2], DType.uint64);
        v1d[u64Idx] = 5.0;
        expect(v1d.toList(), [5.0, 5.0, 11.0, 3.0]);
        final negU64 = NDArray.fromList([-1], [1], DType.uint64);
        expect(() => v1d[negU64] = 1.0, throwsRangeError);
      });
    });

    test(
      'scalar wrapping, comparisons, bitwise/shift specs, squeeze, moveaxis',
      () {
        NDArray.scope(() {
          final u16 = NDArray.fromList([1, 2], [2], DType.uint16);
          final u32 = NDArray.fromList([1, 2], [2], DType.uint32);
          final i64 = NDArray.fromList([1, 2], [2], DType.int64);
          final u64 = NDArray.fromList([1, 2], [2], DType.uint64);
          final bf16 = NDArray.fromList([1.0, 2.0], [2], DType.bfloat16);
          final b = NDArray.fromList([true, false], [2], DType.boolean);
          final f32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
          final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);

          expect((u16 + 1).toList(), [2, 3]);
          expect((u32 + 1).toList(), [2, 3]);
          expect((i64 + 1).toList(), [2, 3]);
          expect((u64 + 1).toList(), [2, 3]);
          expect((bf16 + 1.0).toList(), [2.0, 3.0]);
          expect((b & true).toList(), [true, false]);
          expect((f64 + double.nan).getCell([0]).isNaN, isTrue);
          expect((f64 + double.infinity).getCell([0]), double.infinity);

          expect(() => f32 + 1e100, throwsArgumentError);
          expect(() => i64 + 1.5, throwsArgumentError);
          expect(() => i64 + true, throwsArgumentError);
          expect(() => i64 + 'bad', throwsArgumentError);

          final i8 = NDArray.fromList([1, 2], [2], DType.int8);
          expect((i8 >= 1000).toList(), [false, false]);
          expect((i8 >= 1).toList(), [true, true]);
          expect((i8 <= 1000).toList(), [true, true]);
          expect((i8.ne(1000)).toList(), [true, true]);
          expect((i8.ne(1)).toList(), [false, true]);
          i8.dispose();
          expect(() => i8 > 1000, throwsStateError);

          final bw = u16.asBitwiseDType;
          expect((bw & bw).toList(), [1, 2]);
          expect((bw | bw).toList(), [1, 2]);
          expect((bw ^ bw).toList(), [0, 0]);
          expect((~bw).shape, [2]);

          final sh = u16.asIntegerDType;
          expect((sh << 1).toList(), [2, 4]);
          expect((sh >> 1).toList(), [0, 1]);

          final NDArray<DTypeTag> base = f64;
          expect((base / 2.0).toList(), [0.5, 1.0]);
          expect(base.data.length, 2);
          expect(base.toList(), [1.0, 2.0]);
          expect(base.getCell([0]), 1.0);
          base.setCell([0], 3.0);
          expect(base.getCellRaw(0), 3.0);
          base.setCellRaw(0, 4.0);
          expect(base.getCellFlat(0), 4.0);
          base.setCellFlat(0, 5.0);
          base.fill(6.0);
          expect(base.toList(), [6.0, 6.0]);
          final NDArray<DTypeTag> base0d = NDArray.scalar(
            9.0,
            dtype: DType.float64,
          );
          expect(base0d.scalar, 9.0);

          final m = NDArray.zeros([2, 1, 3], DType.float64);
          expect(() => m.squeeze(axis: [99]), throwsRangeError);
          expect(() => m.squeeze(axis: [0]), throwsArgumentError);
          expect(() => m.squeeze(axis: 'bad'), throwsArgumentError);
          expect(() => m.moveaxis(0, [0]), throwsArgumentError);
          expect(() => m.moveaxis(99, 0), throwsRangeError);
          expect(() => m.moveaxis(0, 99), throwsRangeError);
        });
      },
    );

    test(
      'equals, contentHashCode, noSuchMethod, toString, Complex & lists',
      () {
        NDArray.scope(() {
          final f64 = NDArray.fromList(
            [double.nan, double.infinity, double.negativeInfinity, -0.0],
            [4],
            DType.float64,
          );
          final i64 = NDArray.zeros([4], DType.int64);
          expect(f64.equals('not array'), isFalse);
          expect(f64.equals(i64), isFalse);
          expect(
            NDArray.zeros([
              0,
            ], DType.float64).equals(NDArray.zeros([0], DType.float64)),
            isTrue,
          );

          final f16Scalar = NDArray.scalar(1.0, dtype: DType.float16);
          expect(f16Scalar.contentHashCode, isA<int>());
          final f16 = NDArray.fromList(
            [-0.0, double.nan, 1.0, 2.0],
            [2, 2],
            DType.float16,
          );
          expect(f16.transposed.contentHashCode, isA<int>());
          final bf16 = NDArray.fromList(
            [-0.0, double.nan, 1.0, 2.0],
            [2, 2],
            DType.bfloat16,
          );
          expect(bf16.transposed.contentHashCode, isA<int>());

          final dynamic d = NDArray.fromList([4.0, 2.0], [2], DType.float64);
          expect((d + 1.0).toList(), [5.0, 3.0]);
          expect((d - 1.0).toList(), [3.0, 1.0]);
          expect((d * 2.0).toList(), [8.0, 4.0]);
          expect((d / 2.0).toList(), [2.0, 1.0]);
          expect((d ~/ 2.0).toList(), [2.0, 1.0]);
          expect((d % 3.0).toList(), [1.0, 2.0]);
          expect(d.getCell([0]), 4.0);
          d.setCell([0], 9.0);
          expect(d.getCellFlat(0), 9.0);
          d.setCellFlat(0, 8.0);
          expect(d.getCellRaw(0), 8.0);
          d.setCellRaw(0, 7.0);
          d.fill(1.0);
          expect(d.toList(), [1.0, 1.0]);
          expect(d.data.length, 2);
          final dynamic d0 = NDArray.scalar(3.0, dtype: DType.float64);
          expect(d0.scalar, 3.0);
          expect(() => d.nonExistentMethod(), throwsNoSuchMethodError);

          expect(f64.toString(), contains('nan'));
          expect(
            NDArray.fromList(
              [const Complex(1, 2), const Complex(3, -4)],
              [2],
              DType.complex128,
            ).toString(),
            contains('j'),
          );
          expect(
            NDArray.zeros([0, 2, 2], DType.float64).toString(),
            contains('shape=[0, 2, 2]'),
          );
          expect(
            NDArray.zeros([2, 2, 2], DType.float64).toString(),
            contains('['),
          );
          expect(
            NDArray.zeros([8, 2, 2], DType.float64).toString(),
            contains('...'),
          );

          expect(listEquals<int>(null, [1]), isFalse);
          expect(listEquals<int>([1], null), isFalse);

          const c = Complex(1.0, 2.0);
          expect(c + 3, const Complex(4.0, 2.0));
          expect(() => c + 'bad', throwsArgumentError);
          expect(c - 1, const Complex(0.0, 2.0));
          expect(() => c - 'bad', throwsArgumentError);
          expect(() => c * 'bad', throwsArgumentError);
          expect((c / const Complex(0.0, 0.0)).real.isInfinite, isTrue);
          expect(() => c / 'bad', throwsArgumentError);
          expect(c.pow(0), const Complex(1.0, 0.0));
          expect(() => c.pow('bad'), throwsArgumentError);

          final cl = ComplexList([1.0, 2.0, 3.0, 4.0]);
          expect(() => cl.length = 1, throwsUnsupportedError);
          expect(cl.getReal(0), 1.0);
          expect(cl.getImag(0), 2.0);
          cl.setRealImag(0, 5.0, 6.0);
          expect(cl[0], const Complex(5.0, 6.0));

          final bl = BoolList([1, 0]);
          expect(() => bl.length = 1, throwsUnsupportedError);
        });
      },
    );
  });

  group('Indexing Operations Branch Coverage', () {
    test('take_along_axis & put_along_axis validation and edge cases', () {
      NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
        final idx = NDArray.fromList([0, 2], [2], DType.int64);
        final disp = NDArray.zeros([2], DType.float64)..dispose();
        final s0 = NDArray.scalar(1.0, dtype: DType.float64);
        final s0Idx = NDArray.scalar(0, dtype: DType.int64);

        expect(() => take_along_axis(disp, idx, 0), throwsStateError);
        expect(() => take_along_axis(a, disp, 0), throwsStateError);
        expect(() => take_along_axis(a, idx, 0, out: disp), throwsStateError);
        expect(() => take_along_axis(s0, s0Idx, 0), throwsArgumentError);
        expect(
          () => take_along_axis(
            NDArray.zeros([2, 3], DType.float64),
            NDArray.zeros([2, 2], DType.int64),
            0,
          ),
          throwsArgumentError,
        );
        final badDTypeOut = NDArray.zeros([2], DType.int64);
        expect(
          () =>
              take_along_axis<DTypeTag, DTypeTag>(a, idx, 0, out: badDTypeOut),
          throwsArgumentError,
        );
        expect(
          () => take_along_axis(
            a,
            idx,
            0,
            out: NDArray.zeros([3], DType.float64),
          ),
          throwsArgumentError,
        );

        expect(() => put_along_axis(disp, idx, 1.0, 0), throwsStateError);
        expect(() => put_along_axis(a, disp, 1.0, 0), throwsStateError);
        expect(() => put_along_axis(a, idx, disp, 0), throwsStateError);
        expect(
          () => put_along_axis(a, idx, 1.0, 0, out: disp),
          throwsStateError,
        );
        expect(() => put_along_axis(s0, s0Idx, 1.0, 0), throwsArgumentError);
        expect(
          () => put_along_axis(a, NDArray.zeros([1, 1], DType.int64), 1.0, 0),
          throwsArgumentError,
        );
        expect(() => put_along_axis(a, idx, 1.0, 5), throwsRangeError);
        expect(
          () => put_along_axis(a, idx, NDArray.zeros([1, 2], DType.float64), 0),
          throwsArgumentError,
        );
        expect(
          () => put_along_axis<DTypeTag, DTypeTag>(
            a,
            idx,
            1.0,
            0,
            out: badDTypeOut,
          ),
          throwsArgumentError,
        );
        expect(
          () => put_along_axis(
            a,
            idx,
            1.0,
            0,
            out: NDArray.zeros([2], DType.float64),
          ),
          throwsArgumentError,
        );
        final ro = a.copy()..isWriteable = false;
        expect(() => put_along_axis(ro, idx, 1.0, 0), throwsStateError);
        expect(() => put_along_axis(a, idx, 1.0, 0, out: ro), throwsStateError);

        // Aliasing in put_along_axis
        final iArr = NDArray.fromList([1, 0, 2], [3], DType.int64);
        put_along_axis(iArr, iArr, 99, 0);
        expect(iArr.toList(), [99, 99, 99]);
      });
    });

    test('choose & select across all dtypes, modes, fast and strided paths', () {
      NDArray.scope(() {
        final disp = NDArray.zeros([2], DType.int64)..dispose();
        final idx = NDArray.fromList([0, 1], [2], DType.int64);
        expect(() => choose(disp, [1, 2]), throwsStateError);
        expect(() => choose(idx, [1, 2], out: disp), throwsStateError);
        expect(() => choose(idx, [disp, 2]), throwsStateError);

        // Scalar int fitting across all integer dtypes + bool/Complex/float
        for (final dt in [
          DType.int8,
          DType.uint8,
          DType.int16,
          DType.uint16,
          DType.int32,
          DType.uint32,
          DType.int64,
          DType.uint64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
          DType.complex128,
          DType.complex64,
          DType.boolean,
        ]) {
          final c0 = NDArray.ones([2], dt);
          final res = choose(idx, [c0, dt == DType.boolean ? false : 1]);
          expect(res.shape, [2]);
        }
        expect(
          choose(NDArray.zeros([1], DType.int64), [true]).dtype,
          DType.boolean,
        );
        expect(choose(idx, [true, false]).dtype, DType.uint8);
        expect(
          choose(idx, [const Complex(1, 0), const Complex(2, 0)]).dtype,
          DType.complex128,
        );

        // Fast path across all index dtypes (int16, int8, uint64, uint32, uint16, uint8)
        for (final idxDt in [
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final aIdx = NDArray.fromList([0, 1], [2], idxDt);
          final aScalar = NDArray.fromList([1], [1], idxDt);
          expect(choose(aIdx, [10, 20]).toList(), [10, 20]);
          expect(choose(aScalar, [10, 20]).toList(), [20]);
        }
        expect(
          () => choose(NDArray.fromList([0.0], [1], DType.float64), [1, 2]),
          throwsUnsupportedError,
        );

        // Wrap and clip with negative indices in fast path and strided slow path
        final negIdx = NDArray.fromList([-1, 3], [2], DType.int64);
        expect(choose(negIdx, [10.0, 20.0], mode: ChooseMode.wrap).toList(), [
          20.0,
          20.0,
        ]);
        expect(choose(negIdx, [10.0, 20.0], mode: ChooseMode.clip).toList(), [
          10.0,
          20.0,
        ]);
        expect(
          () => choose(negIdx, [10.0, 20.0], mode: ChooseMode.raise),
          throwsRangeError,
        );

        // Strided slow path in choose (including uint64 negative bit-pattern)
        final stridedIdx = NDArray.fromList(
          [-1, 0, 5, 0],
          [4],
          DType.int64,
        ).slice([const Slice(step: 2)]);
        expect(choose(stridedIdx, [10, 20], mode: ChooseMode.wrap).toList(), [
          20,
          20,
        ]);
        expect(choose(stridedIdx, [10, 20], mode: ChooseMode.clip).toList(), [
          10,
          20,
        ]);
        expect(
          () => choose(stridedIdx, [10, 20], mode: ChooseMode.raise),
          throwsRangeError,
        );
        final stridedU64 = NDArray.fromList(
          [-1, 0, -2, 0],
          [4],
          DType.uint64,
        ).slice([const Slice(step: 2)]);
        expect(choose(stridedU64, [10, 20], mode: ChooseMode.wrap).shape, [2]);
        expect(choose(stridedU64, [10, 20], mode: ChooseMode.clip).toList(), [
          20,
          20,
        ]);

        expect(
          () => choose(idx, [
            10,
            20,
          ], out: NDArray.zeros([2], DType.float64).asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => choose(idx, [
            10,
            20,
          ], out: NDArray.zeros([3], DType.int64).asAnySpec),
          throwsArgumentError,
        );

        // select validation & edge cases
        final cond = NDArray.fromList([true, false], [2], DType.boolean);
        final dispBool = NDArray.zeros([2], DType.boolean)..dispose();
        expect(() => select([cond], [1], out: disp), throwsStateError);
        expect(() => select([dispBool], [1]), throwsStateError);
        expect(() => select([cond], [disp]), throwsStateError);
        expect(() => select([cond], [1], defaultValue: disp), throwsStateError);
        expect(select([cond], [true]).dtype, DType.boolean);
        expect(select([cond], [const Complex(1, 2)]).dtype, DType.complex128);
        expect(
          () => select(
            [cond],
            [1],
            out: NDArray.zeros([2], DType.float64).asAnySpec,
          ),
          throwsArgumentError,
        );
        expect(
          () => select(
            [cond],
            [1],
            out: NDArray.zeros([3], DType.int64).asAnySpec,
          ),
          throwsArgumentError,
        );
      });
    });

    test(
      'unravel_index, ravel_multi_index, indices, sparse/diag/tril/triu/maskIndices',
      () {
        NDArray.scope(() {
          final idx = NDArray.fromList([1, 3], [2], DType.int64);
          final disp = NDArray.zeros([2], DType.int64)..dispose();
          expect(() => unravelIndex(disp, [2, 2]), throwsStateError);
          expect(
            () => unravelIndex(NDArray.zeros([2], DType.float64), [2, 2]),
            throwsArgumentError,
          );
          expect(() => unravelIndex(idx, []), throwsArgumentError);
          expect(() => unravelIndex(idx, [2, -1]), throwsArgumentError);
          expect(
            () => unravelIndex(idx, [2, 2], out: [idx]),
            throwsArgumentError,
          );
          expect(
            () => unravelIndex(idx, [2, 2], out: [disp, idx]),
            throwsStateError,
          );
          expect(
            () => unravelIndex(
              idx,
              [2, 2],
              out: [
                NDArray.zeros([3], DType.int64),
                idx,
              ],
            ),
            throwsArgumentError,
          );
          // out[0] and out[1] sharing memory
          final sharedOut = NDArray.zeros([2], DType.int64);
          final uRes = unravelIndex(idx, [2, 2], out: [sharedOut, sharedOut]);
          expect(uRes.length, 2);

          // ravel_multi_index validation & edge cases
          expect(() => ravelMultiIndex([], [2]), throwsArgumentError);
          expect(() => ravelMultiIndex([idx], [2, 2]), throwsArgumentError);
          expect(() => ravelMultiIndex([disp], [2]), throwsStateError);
          expect(
            () => ravelMultiIndex(
              [
                NDArray.zeros([2], DType.float64),
              ],
              [2],
            ),
            throwsArgumentError,
          );
          expect(() => ravelMultiIndex([idx], [0]), throwsArgumentError);
          expect(
            () => ravelMultiIndex([idx, idx], [0x4000000000000000, 4]),
            throwsArgumentError,
          );
          expect(
            () => ravelMultiIndex([idx, idx], [2, 2], mode: [ChooseMode.wrap]),
            throwsArgumentError,
          );
          expect(
            () => ravelMultiIndex([idx, idx], [2, 2], mode: 'bad'),
            throwsArgumentError,
          );
          expect(
            () => ravelMultiIndex([idx], [4], out: disp),
            throwsStateError,
          );
          expect(
            () => ravelMultiIndex(
              [idx],
              [4],
              out: NDArray.zeros([3], DType.int64),
            ),
            throwsArgumentError,
          );
          final rOut = NDArray.zeros([2], DType.int64);
          expect(ravelMultiIndex([rOut], [4], out: rOut), same(rOut));

          // indices, sparseIndices, diagIndices, trilIndices, triuIndices, maskIndices
          expect(() => indices([-1]), throwsArgumentError);
          expect(() => indices([2], out: disp.asAnySpec), throwsStateError);
          expect(
            () => indices(
              [2],
              dtype: DType.int64.asAnySpec,
              out: NDArray.zeros([1, 2], DType.float64).asAnySpec,
            ),
            throwsArgumentError,
          );
          expect(
            () => indices([2], out: NDArray.zeros([2], DType.int64).asAnySpec),
            throwsArgumentError,
          );
          final stridedOut = NDArray.zeros([
            1,
            4,
          ], DType.int64).slice([const Slice.all(), const Slice(step: 2)]);
          expect(indices([2], out: stridedOut.asAnySpec).shape, [1, 2]);
          final f64GridOut = NDArray.zeros([1, 2], DType.float64);
          expect(
            indices(
              [2],
              dtype: DType.float64.asAnySpec,
              out: f64GridOut.asAnySpec,
            ),
            same(f64GridOut),
          );

          expect(() => sparseIndices([-1]), throwsArgumentError);
          expect(sparseIndices([2, 3]).length, 2);

          expect(() => diagIndices(-1), throwsArgumentError);
          expect(() => diagIndices(2, ndim: 0), throwsArgumentError);
          expect(diagIndices(2).length, 2);
          expect(() => diagIndicesFrom(disp), throwsStateError);
          expect(() => diagIndicesFrom(idx), throwsArgumentError);
          expect(
            diagIndicesFrom(NDArray.zeros([2, 2], DType.float64)).length,
            2,
          );

          expect(trilIndices(0).row.size, 0);
          expect(() => trilIndices(-1), throwsArgumentError);
          expect(() => trilIndices(2, m: -1), throwsArgumentError);
          expect(trilIndices(2).row.size, 3);
          expect(() => trilIndicesFrom(disp), throwsStateError);
          expect(() => trilIndicesFrom(idx), throwsArgumentError);
          expect(
            trilIndicesFrom(NDArray.zeros([2, 2], DType.float64)).row.size,
            3,
          );

          expect(() => triuIndices(-1), throwsArgumentError);
          expect(() => triuIndices(2, m: -1), throwsArgumentError);
          expect(triuIndices(2).row.size, 3);
          expect(() => triuIndicesFrom(disp), throwsStateError);
          expect(() => triuIndicesFrom(idx), throwsArgumentError);
          expect(
            triuIndicesFrom(NDArray.zeros([2, 2], DType.float64)).row.size,
            3,
          );

          expect(() => maskIndices(-1, triu), throwsArgumentError);
          expect(
            () => maskIndices(2, (m, {int k = 0}) => disp),
            throwsStateError,
          );
          expect(
            () => maskIndices(2, (m, {int k = 0}) => idx),
            throwsArgumentError,
          );
          expect(maskIndices(2, triu).row.size, 3);

          // Additional indexing branches (uint64 ravel_multi_index, aliasing, broadcast errors)
          final u64Neg = NDArray.fromList([-1, 0], [2], DType.uint64);
          expect(
            () => ravelMultiIndex([u64Neg], [3], mode: ChooseMode.raise),
            throwsRangeError,
          );
          expect(
            ravelMultiIndex([u64Neg], [3], mode: ChooseMode.clip).toList(),
            [2, 0],
          );
          expect(
            ravelMultiIndex([u64Neg], [3], mode: ChooseMode.wrap).toList(),
            [0, 0],
          );
          expect(
            () => choose(idx, [
              NDArray.zeros([3], DType.int64),
            ]),
            throwsArgumentError,
          );
          final cOutChoice = NDArray.fromList([10, 20], [2], DType.int64);
          expect(
            choose(NDArray.fromList([0, 1], [2], DType.int64), [
              cOutChoice,
              30,
            ], out: cOutChoice),
            same(cOutChoice),
          );
          final cond2 = NDArray.fromList([true, false], [2], DType.boolean);
          final sOut = NDArray.fromList([5, 6], [2], DType.int64);
          expect(select([cond2], [sOut], out: sOut), same(sOut));
          final sOutNoAlias = NDArray.zeros([2], DType.int64);
          expect(select([cond2], [1], out: sOutNoAlias), same(sOutNoAlias));

          final tArr = NDArray.fromList([10.0, 20.0], [2], DType.float64);
          final tIdx = NDArray.fromList([1, 0], [2], DType.int64);
          expect(take_along_axis(tArr, tIdx, 0, out: tArr), same(tArr));
          expect(take_along_axis(tIdx, tIdx, 0, out: tIdx), same(tIdx));
          expect(
            () => put_along_axis(
              tArr,
              tIdx,
              NDArray.zeros([3], DType.float64),
              0,
            ),
            throwsArgumentError,
          );
        });
      },
    );

    test('additional ndarray core & utility branches', () {
      expect(uint64Compare(1, 1), 0);
      expect(uint64Compare(-1, 1), 1);
      expect(uint64Compare(1, -1), -1);
      expect(uint64Compare(2, 1), greaterThan(0));

      expect(DType.float16.isHalf, isTrue);
      expect(DType.bfloat16.isHalf, isTrue);
      expect(DType.float64.isHalf, isFalse);
      for (final dt in DType.values) {
        expect(dt.isUnsigned, isA<bool>());
        expect(dt.isSignedInteger, isA<bool>());
      }

      expect(NDArray.trackedAllocations, isA<List<NDArray<DTypeTag>>>());
      expect(NDArray.checkNoLeaks(), isA<bool>());
      NDArray.clearTrackedAllocations();

      final ret = NDArray.returning(
        () => NDArray.fromList([1, 2], [2], DType.int64),
      );
      expect(ret.toList(), [1, 2]);
      ret.dispose();

      final unm = NDArray.unmanaged(
        () => NDArray.fromList([3, 4], [2], DType.int64),
      );
      expect(unm.toList(), [3, 4]);
      unm.dispose();

      NDArray.scope(() {
        expect(
          () => NDArray.create([3], DType.uint8, strides: [0x4000000000000000]),
          throwsArgumentError,
        );
        expect(
          () =>
              NDArray.create([2], DType.uint8, strides: [-0x8000000000000000]),
          throwsArgumentError,
        );
        expect(
          () => NDArray.fromList([1], [2], DType.int64),
          throwsArgumentError,
        );
        expect(
          () => NDArray.fromList([3000000000], [1], DType.int32),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-3000000000], [1], DType.int32),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([40000], [1], DType.int16),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-40000], [1], DType.int16),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([200], [1], DType.int8),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-200], [1], DType.int8),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-1], [1], DType.uint32),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([5000000000], [1], DType.uint32),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-1], [1], DType.uint16),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([70000], [1], DType.uint16),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-1], [1], DType.uint8),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([300], [1], DType.uint8),
          throwsRangeError,
        );

        for (final dt in DType.values) {
          final v = dt == DType.boolean
              ? true
              : (dt.isComplex ? const Complex(1, 2) : 1);
          final f = NDArray.full([2], v, dtype: dt);
          f.fill(v);
          expect(f.shape, [2]);
          if (dt.isInteger || dt.isFloating || dt.isComplex) {
            expect(((f as dynamic) + 1).shape, [2]);
          }
        }

        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final sendCopy = a.toSendable();
        expect(sendCopy.materialize().toList(), [1.0, 2.0, 3.0, 4.0]);
        final sendBorrow = a.toSendableBorrow();
        expect(sendBorrow.materialize().toList(), [1.0, 2.0, 3.0, 4.0]);

        expect(a.expandDims(0).shape, [1, 2, 2]);
        expect(a.expandDims(2).shape, [2, 2, 1]);
        expect(() => a.expandDims(5), throwsRangeError);
        final sq = a.expandDims(0);
        expect(sq.squeeze().shape, [2, 2]);
        expect(sq.squeeze(axis: 0).shape, [2, 2]);
        expect(() => sq.squeeze(axis: 5), throwsRangeError);
        expect(() => sq.squeeze(axis: 1), throwsArgumentError);

        expect(a.swapaxes(0, 0), same(a));
        expect(a.swapaxes(0, 1).shape, [2, 2]);
        expect(() => a.swapaxes(5, 0), throwsRangeError);
        expect(() => a.swapaxes(0, 5), throwsRangeError);

        expect(a.moveaxis(0, 1).shape, [2, 2]);
        expect(() => a.moveaxis([0], [0, 1]), throwsArgumentError);
        expect(() => a.moveaxis([0, 0], [0, 1]), throwsArgumentError);
        expect(() => a.moveaxis([0, 1], [0, 0]), throwsArgumentError);

        // Slicing with negative step
        final s1d = NDArray.fromList([1, 2, 3, 4], [4], DType.int64);
        expect(s1d[const Slice.all(step: -1)].toList(), [4, 3, 2, 1]);
        expect(s1d[const Slice(start: 3, stop: 0, step: -1)].toList(), [
          4,
          3,
          2,
        ]);
        expect(s1d[const Slice(start: -1, stop: -4, step: -1)].toList(), [
          4,
          3,
          2,
        ]);
        expect(s1d[const Slice(start: 1, stop: 3, step: -1)].size, 0);

        // Bitwise & shift operators
        final bw = NDArray.fromList([3, 5], [2], DType.int64);
        expect((bw | 1).toList(), [3, 5]);
        expect((bw ^ 1).toList(), [2, 4]);
        expect((~bw).toList(), [-4, -6]);
        expect((bw << 1).toList(), [6, 10]);
        expect((bw >> 1).toList(), [1, 2]);
        expect((bw / 2).toList(), [1.5, 2.5]);

        // toString formatting
        expect(
          NDArray.scalar(42, dtype: DType.int64).toString(),
          contains('42'),
        );
        expect(NDArray.zeros([0], DType.int64).toString(), contains('[]'));
        expect(NDArray.zeros([8], DType.int64).toString(), contains('...'));
        expect(
          NDArray.zeros([0, 2], DType.int64).toString(),
          contains('shape=[0, 2]'),
        );
        expect(NDArray.zeros([8, 8], DType.int64).toString(), contains('...'));

        // Complex methods
        const c1 = Complex(3.0, 4.0);
        const c2 = Complex(1.0, 2.0);
        expect(c1 + c2, const Complex(4.0, 6.0));
        expect(c1 - c2, const Complex(2.0, 2.0));
        expect(-c1, const Complex(-3.0, -4.0));
        expect(c1.abs, 5.0);
        expect(c1.arg, isA<double>());
        expect(c1.log(), isA<Complex>());
        expect(c1.pow(const Complex(0.0, 0.0)), const Complex(1.0, 0.0));
        expect(c1.pow(c2), isA<Complex>());

        final cl2 = ComplexList([1.0, 2.0]);
        cl2[0] = const Complex(7.0, 8.0);
        expect(cl2.length, 1);
        expect(cl2[0], const Complex(7.0, 8.0));

        final bl2 = BoolList([1, 0]);
        expect(bl2.backingList, [1, 0]);
        expect(bl2.length, 2);
        expect(bl2[0], isTrue);
        bl2[1] = true;
        expect(bl2[1], isTrue);

        // Remaining core NDArray branches
        final disp = NDArray.zeros([2, 2], DType.float64)..dispose();
        final disp0 = NDArray.scalar(1.0, dtype: DType.float64)..dispose();
        final dispBool = NDArray.zeros([2], DType.boolean)..dispose();
        final base = NDArray.zeros([1], DType.float64);
        final vBcast = NDArray.view(base, shape: [3], strides: [0]);
        expect(vBcast.isWriteable, isFalse);
        expect(() => vBcast.isWriteable = true, throwsArgumentError);
        final roBase = NDArray.zeros([2], DType.float64)..isWriteable = false;
        final roView = roBase.slice([const Slice.all()]);
        expect(() => roView.isWriteable = true, throwsArgumentError);

        expect(() => NDArray.zeros([-1], DType.float64), throwsArgumentError);
        expect(() => disp.data, throwsStateError);
        expect(
          () => NDArray.create([0x2000000000000000], DType.float64),
          throwsArgumentError,
        );
        expect(
          () => NDArray.computeCStrides([2, 0x4000000000000000, 4]),
          throwsArgumentError,
        );
        final ptr = calloc<ffi.Uint8>(16);
        try {
          expect(
            () => NDArray.fromPointer(ptr.cast(), [
              0x2000000000000000,
            ], DType.float64),
            throwsArgumentError,
          );
        } finally {
          calloc.free(ptr);
        }

        expect(() => disp.pointer, throwsStateError);
        expect(() => disp.reshape([4]), throwsStateError);
        expect(() => a.reshape([5]), throwsArgumentError);
        expect(a.reshape([2, 2]).shape, [2, 2]);
        expect(() => disp.copy(), throwsStateError);
        expect(
          () => a.copy(
            out: NDArray.zeros([2, 2], DType.float64)..isWriteable = false,
          ),
          throwsArgumentError,
        );
        expect(
          () => a.copy(out: NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );
        expect(a.transposed.copy(out: a), same(a));
        expect(() => disp.astype(DType.int64), throwsStateError);
        expect(a.astype(DType.float64, copy: false), same(a));
        NDArray.scalar(
          1.0,
          dtype: DType.float64,
        ).copyToContiguous(NDArray.scalar(0.0, dtype: DType.float64));
        expect(() => disp.ravel(), throwsStateError);
        expect(() => roBase.fill(1.0), throwsStateError);

        for (final dt in DType.values) {
          final v = dt == DType.boolean
              ? true
              : (dt.isComplex ? const Complex(1, 2) : 1);
          NDArray.zeros([2, 2], dt).transposed.fill(v);
        }

        expect(() => disp0.scalar, throwsStateError);
        expect(() => disp.getCell([0, 0]), throwsStateError);
        expect(a.getCell([-1, -1]), isA<double>());
        expect(() => disp.setCell([0, 0], 1.0), throwsStateError);
        expect(() => roBase.setCell([0], 1.0), throwsStateError);
        a.setCell([-1, -1], 5.0);
        expect(() => roBase.setCellRaw(0, 1.0), throwsStateError);
        expect(() => roBase.setCellFlat(0, 1.0), throwsStateError);

        final bMask = NDArray.fromList([true, false], [2], DType.boolean);
        final iIdx = NDArray.fromList([0], [1], DType.int64);
        final v1 = NDArray.fromList([1.0], [1], DType.float64);
        final selfArr = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => disp.setByMask(bMask, v1), throwsStateError);
        expect(() => roBase.setByMask(bMask, v1), throwsStateError);
        selfArr.setByMask(
          bMask,
          selfArr.slice([const Slice(start: 0, stop: 1)]),
        );
        expect(() => roBase.setByMaskScalar(bMask, disp), throwsStateError);
        expect(() => roBase.setByMaskScalar(bMask, 1.0), throwsStateError);
        expect(() => roBase.setIndicesScalar(iIdx, disp), throwsStateError);
        expect(() => roBase.setIndicesScalar(iIdx, 1.0), throwsStateError);
        expect(() => disp.setIndices(iIdx, v1), throwsStateError);
        expect(() => roBase.setIndices(iIdx, v1), throwsStateError);
        selfArr.setIndices(NDArray.fromList([-1], [1], DType.int64), v1);

        expect(() => selfArr[0] = disp0, throwsStateError);
        final c64Arr = NDArray.zeros([2], DType.complex64);
        c64Arr[0] = const Complex(1, 2);
        expect(
          selfArr[NDArray.fromList([0, 1], [2], DType.int64)].toList(),
          isNotEmpty,
        );
        expect(
          () => selfArr[NDArray.fromList([-1], [1], DType.uint64)],
          throwsRangeError,
        );
        expect(() => selfArr[[disp]], throwsStateError);
        expect(() => roBase[const Slice.all()] = 1.0, throwsStateError);
        selfArr[const Slice.all()] = selfArr;
        expect(() => a[[const Slice.all(), 99]] = 1.0, throwsRangeError);
        expect(
          () =>
              a[[
                    const Slice.all(),
                    Indices([99]),
                  ]] =
                  1.0,
          throwsRangeError,
        );
        a[[const Slice.all(), const Slice.all()]] = a.copy();
        a[[const Slice.all(), const Slice.all()]] = NDArray.fromList(
          [1],
          [1],
          DType.int64,
        );
        a[[
          0,
          Indices([0, 1]),
        ]] = NDArray.fromList(
          [9.0],
          [1],
          DType.float64,
        );

        final dispInt = NDArray.zeros([1], DType.int64)..dispose();
        expect(selfArr[[0, 1]].toList(), isNotEmpty);
        expect(() => selfArr[dispBool], throwsStateError);
        expect(() => selfArr[dispInt], throwsStateError);
        expect(a[NDArray.fromList([0, 1], [1, 2], DType.int64)].shape, [
          1,
          2,
          2,
        ]);
        expect(() => roBase[0] = 1.0, throwsStateError);
        expect(
          () => selfArr[[0]] = NDArray.zeros([2], DType.float64),
          throwsArgumentError,
        );
        expect(() => selfArr[dispBool] = 1.0, throwsStateError);
        selfArr[bMask] = v1;
        expect(() => selfArr[dispInt] = 1.0, throwsStateError);
        a[NDArray.fromList([0, 1], [1, 2], DType.int64)] = NDArray.fromList(
          [1.0],
          [1],
          DType.float64,
        );

        expect(
          () =>
              (selfArr as NDArray<DTypeTag>) + NDArray.zeros([2], DType.int64),
          throwsArgumentError,
        );
        expect((-selfArr).shape, [2]);
        final i8Arr = NDArray.fromList([1, 2], [2], DType.int8);
        expect((i8Arr > 999).toList(), [false, false]);
        expect((i8Arr < 999).toList(), [true, true]);
        expect((i8Arr <= 1).toList(), [true, false]);
        expect(i8Arr.eq(999).toList(), [false, false]);
        expect(
          a[[
                0,
                Indices([0, 1]),
              ]]
              .shape,
          [2],
        );
        expect(() => selfArr.take([0], axis: 5), throwsRangeError);
        expect(() => disp.applyMask(bMask), throwsStateError);
        final bMask2d = NDArray.fromList(
          [true, false, false, true],
          [2, 2],
          DType.boolean,
        );
        expect(a.transposed.applyMask(bMask2d.transposed).size, 2);
        expect(() => disp.toList(), throwsStateError);
        expect(() => disp.expandDims(0), throwsStateError);
        expect(() => disp.squeeze(), throwsStateError);

        expect(selfArr.equals(selfArr), isTrue);
        expect(i8Arr.equals(i8Arr.copy()), isTrue);
        expect(
          NDArray.zeros([2, 2], DType.int16).transposed.contentHashCode,
          isA<int>(),
        );
        expect(
          NDArray.zeros([2, 2], DType.int8).transposed.contentHashCode,
          isA<int>(),
        );
        expect(disp.toString(), contains('disposed'));
        expect(c1.toString(), contains('i'));
        expect(c1.hashCode, isA<int>());
        expect(
          NDArray.zeros([0, 2], DType.float64)[[
                const Slice.all(),
                Indices([0]),
              ]]
              .size,
          0,
        );
        selfArr.setCellRaw(0, 3.0);
        final selfSlice = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        selfSlice[[
              Indices([1, 0]),
            ]] =
            selfSlice;
        expect(selfSlice.toList(), [20.0, 10.0]);
      });
    });
  });
}
