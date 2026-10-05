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

import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('1. Core NDArray Constructors & Properties Across All 15 DTypes', () {
    test('NDArray.zeros across all 15 DTypes', () {
      NDArray.scope(() {
        for (final dt in DType.values) {
          final a = NDArray.zeros([2, 3], dt);
          expect(a.shape, equals([2, 3]));
          expect(a.dtype, equals(dt));
          expect(a.size, equals(6));
          expect(a.rank, equals(2));
          expect(a.isContiguous, isTrue);

          if (dt == DType.boolean) {
            expect(a.getCell([0, 0]), isFalse);
          } else if (dt.isComplex) {
            final c = a.getCell([0, 0]) as Complex;
            expect(c.real, equals(0.0));
            expect(c.imag, equals(0.0));
          } else {
            expect((a.getCell([0, 0]) as num).toDouble(), equals(0.0));
          }
        }
      });
    });

    test('NDArray.ones across all 15 DTypes', () {
      NDArray.scope(() {
        for (final dt in DType.values) {
          final a = NDArray.ones([3, 2], dt);
          expect(a.shape, equals([3, 2]));
          expect(a.dtype, equals(dt));
          expect(a.size, equals(6));

          if (dt == DType.boolean) {
            expect(a.getCell([1, 1]), isTrue);
          } else if (dt.isComplex) {
            final c = a.getCell([1, 1]) as Complex;
            expect(c.real, equals(1.0));
            expect(c.imag, equals(0.0));
          } else {
            expect((a.getCell([1, 1]) as num).toDouble(), equals(1.0));
          }
        }
      });
    });

    test(
      'NDArray.create and NDArray.full across numeric, bool & complex types',
      () {
        NDArray.scope(() {
          final a = NDArray.create([2, 2, 2], DType.float64);
          expect(a.shape, equals([2, 2, 2]));
          expect(a.size, equals(8));

          final fullF64 = NDArray.full([2, 2], 42.5, dtype: DType.float64);
          expect(fullF64.getCell([0, 0]), equals(42.5));
          expect(fullF64.getCell([1, 1]), equals(42.5));

          final fullI32 = NDArray.full([3], 7, dtype: DType.int32);
          expect(fullI32.getCell([0]), equals(7));
          expect(fullI32.getCell([2]), equals(7));

          final fullC128 = NDArray.full(
            [2],
            Complex(3.0, -4.0),
            dtype: DType.complex128,
          );
          expect(fullC128.getCell([0]).real, equals(3.0));
          expect(fullC128.getCell([0]).imag, equals(-4.0));

          final fullBool = NDArray.full([2, 2], true, dtype: DType.boolean);
          expect(fullBool.getCell([0, 1]), isTrue);
        });
      },
    );

    test('NDArray.eye and NDArray.arange', () {
      NDArray.scope(() {
        final id3 = NDArray.eye(3, DType.float64);
        expect(id3.shape, equals([3, 3]));
        expect(id3.isSquare, isTrue);
        for (var i = 0; i < 3; i++) {
          for (var j = 0; j < 3; j++) {
            expect(id3.getCell([i, j]), equals(i == j ? 1.0 : 0.0));
          }
        }

        final eyeI32 = NDArray.eye(4, DType.int32);
        expect(eyeI32.shape, equals([4, 4]));
        expect(eyeI32.getCell([2, 2]), equals(1));
        expect(eyeI32.getCell([2, 1]), equals(0));

        final ar1 = NDArray.arange(0.0, 5.0, dtype: DType.int32);
        expect(ar1.shape, equals([5]));
        expect(ar1.toList(), equals([0, 1, 2, 3, 4]));

        final ar2 = NDArray.arange(2.0, 10.0, step: 2.0, dtype: DType.float64);
        expect(ar2.toList(), equals([2.0, 4.0, 6.0, 8.0]));

        expect(
          () => NDArray.arange(0.0, 5.0, step: 0.0, dtype: DType.float64),
          throwsArgumentError,
        );
        expect(
          () => NDArray.arange(5.0, 0.0, step: 1.0, dtype: DType.float64),
          throwsArgumentError,
        );
      });
    });

    test('NDArray.fromList, scalar, view, and fromPointer', () {
      NDArray.scope(() {
        final flat = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int64);
        expect(flat.shape, equals([2, 3]));
        expect(flat.getCell([1, 2]), equals(6));

        final sc = NDArray.scalar(99.0, dtype: DType.float64);
        expect(sc.shape, equals([]));
        expect(sc.rank, equals(0));
        expect(sc.size, equals(1));
        expect(sc.getCell([]), equals(99.0));

        final scC = NDArray.scalar(Complex(1.0, 2.0), dtype: DType.complex128);
        expect(scC.getCell([]).real, equals(1.0));
        expect(scC.getCell([]).imag, equals(2.0));

        // fromPointer
        final ptr = calloc<Double>(4);
        ptr[0] = 10.0;
        ptr[1] = 20.0;
        ptr[2] = 30.0;
        ptr[3] = 40.0;
        final fromPtr = NDArray.fromPointer(ptr.cast<Void>(), [
          2,
          2,
        ], DType.float64);
        expect(fromPtr.shape, equals([2, 2]));
        expect(fromPtr.getCell([0, 1]), equals(20.0));
        expect(fromPtr.getCell([1, 1]), equals(40.0));
      });
    });

    test('Shape comparison, fill, and invalid shape error checks', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 3], DType.float64);
        final b = NDArray.ones([2, 3], DType.int32);
        final c = NDArray.zeros([3, 2], DType.float64);
        expect(a.hasSameShape(b), isTrue);
        expect(a.hasSameShape(c), isFalse);

        a.fill(123.0);
        expect(a.getCell([0, 0]), equals(123.0));
        expect(a.getCell([1, 2]), equals(123.0));

        expect(
          () => NDArray.create([-1, 3], DType.float64),
          throwsArgumentError,
        );
        expect(
          () => NDArray.fromList([1, 2], [3], DType.int32),
          throwsArgumentError,
        );
      });
    });
  });

  group('2. Float16 & BFloat16 Bitwise Utilities & List Wrappers', () {
    test('Float16 IEEE 754 encoding and decoding edge cases', () {
      // Zero and negative zero
      final zeroBits = Float16Utils.encodeFloat16(0.0);
      expect(Float16Utils.decodeFloat16(zeroBits), equals(0.0));
      final negZeroBits = Float16Utils.encodeFloat16(-0.0);
      expect(Float16Utils.decodeFloat16(negZeroBits).isNegative, isTrue);

      // Infinities
      final infBits = Float16Utils.encodeFloat16(double.infinity);
      expect(Float16Utils.decodeFloat16(infBits), equals(double.infinity));
      final negInfBits = Float16Utils.encodeFloat16(double.negativeInfinity);
      expect(
        Float16Utils.decodeFloat16(negInfBits),
        equals(double.negativeInfinity),
      );

      // NaN
      final nanBits = Float16Utils.encodeFloat16(double.nan);
      expect(Float16Utils.decodeFloat16(nanBits).isNaN, isTrue);

      // Normal numbers
      for (final v in [1.0, -1.0, 0.5, 1.5, 2.0, 100.0, 65504.0]) {
        final bits = Float16Utils.encodeFloat16(v);
        expect(Float16Utils.decodeFloat16(bits), closeTo(v, 1e-3));
      }

      // Subnormals
      final subnormalVal =
          5.960464477539063e-8; // smallest positive subnormal in float16
      final subBits = Float16Utils.encodeFloat16(subnormalVal);
      expect(Float16Utils.decodeFloat16(subBits), closeTo(subnormalVal, 1e-12));

      // Overflow to infinity
      final overflowBits = Float16Utils.encodeFloat16(100000.0);
      expect(Float16Utils.decodeFloat16(overflowBits), equals(double.infinity));

      // Underflow to zero
      final underflowBits = Float16Utils.encodeFloat16(1e-12);
      expect(Float16Utils.decodeFloat16(underflowBits), equals(0.0));
    });

    test('BFloat16 encoding and decoding', () {
      final zeroBits = Float16Utils.encodeBFloat16(0.0);
      expect(Float16Utils.decodeBFloat16(zeroBits), equals(0.0));

      final nanBits = Float16Utils.encodeBFloat16(double.nan);
      expect(Float16Utils.decodeBFloat16(nanBits).isNaN, isTrue);

      for (final v in [1.0, -2.5, 3.14159, 1000.0]) {
        final bits = Float16Utils.encodeBFloat16(v);
        expect(Float16Utils.decodeBFloat16(bits), closeTo(v, 0.05));
      }
    });

    test('Float16List & BFloat16List list operations', () {
      final u16Buffer1 = Uint16List(4);
      final f16List = Float16List(u16Buffer1);
      expect(f16List.length, equals(4));
      f16List[0] = 1.0;
      f16List[1] = 2.5;
      f16List[2] = -3.0;
      f16List[3] = 4.0;
      expect(f16List[0], closeTo(1.0, 1e-3));
      expect(f16List[1], closeTo(2.5, 1e-3));
      expect(f16List[2], closeTo(-3.0, 1e-3));
      expect(f16List[3], closeTo(4.0, 1e-3));
      expect(() => f16List.length = 5, throwsUnsupportedError);

      final u16Buffer2 = Uint16List(3);
      final bf16List = BFloat16List(u16Buffer2);
      expect(bf16List.length, equals(3));
      bf16List[0] = 10.0;
      bf16List[1] = 20.0;
      bf16List[2] = 30.0;
      expect(bf16List[0], closeTo(10.0, 0.1));
      expect(bf16List[1], closeTo(20.0, 0.1));
      expect(bf16List[2], closeTo(30.0, 0.1));
      expect(() => bf16List.length = 10, throwsUnsupportedError);
    });
  });

  group('3. Advanced Slicing, Step, Ellipsis, Newaxis & Mask Indexing', () {
    test('1D and 2D slicing with positive and negative steps', () {
      NDArray.scope(() {
        final a = NDArray.arange(0.0, 10.0, dtype: DType.float64); // [0..9]
        final s1 = a.slice([Slice(start: 1, stop: 8, step: 2)]); // [1, 3, 5, 7]
        expect(s1.toList(), equals([1.0, 3.0, 5.0, 7.0]));

        final sRev = a.slice([
          Slice(start: 8, stop: 1, step: -2),
        ]); // [8, 6, 4, 2]
        expect(sRev.toList(), equals([8.0, 6.0, 4.0, 2.0]));

        final sAllRev = a.slice([Slice(step: -1)]);
        expect(
          sAllRev.toList(),
          equals([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        );

        // 2D slice
        final mat = NDArray.fromList(
          [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
          [3, 4],
          DType.int32,
        );

        final subMat = mat.slice([
          Slice(start: 0, stop: 2),
          Slice(start: 1, stop: 3),
        ]);
        expect(subMat.shape, equals([2, 2]));
        expect(subMat.toList(), equals([11, 12, 21, 22]));
      });
    });

    test('Operator [] and []= polymorphic indexing', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);

        // Coordinate access via List<int>
        expect(a[[0, 1]], equals(2));
        expect(a[[1, 2]], equals(6));

        // Single integer row access
        final row0 = a[0] as NDArray<AnySpec>;
        expect(row0.shape, equals([3]));
        expect(row0.toList(), equals([1, 2, 3]));

        // Scalar mutation via operator []=
        a[[0, 1]] = 99;
        expect(a[[0, 1]], equals(99));

        // Sliced mutation via operator []=
        a[1] = NDArray.fromList([77, 88, 99], [3], DType.int32);
        expect(a.toList(), equals([1, 99, 3, 77, 88, 99]));
      });
    });

    test('Boolean mask indexing and assignment', () {
      NDArray.scope(() {
        final a = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);
        final mask = NDArray.fromList(
          [true, false, true, false, true],
          [5],
          DType.boolean,
        );

        final masked = a[mask] as NDArray<AnySpec>;
        expect(masked.shape, equals([3]));
        expect(masked.toList(), equals([10, 30, 50]));

        // Boolean mask assignment
        a[mask] = 0;
        expect(a.toList(), equals([0, 20, 0, 40, 0]));
      });
    });

    test('expandDims and squeeze', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 3], DType.float64);
        final exp0 = a.expandDims(0);
        expect(exp0.shape, equals([1, 2, 3]));

        final exp1 = a.expandDims(1);
        expect(exp1.shape, equals([2, 1, 3]));

        final expEnd = a.expandDims(2);
        expect(expEnd.shape, equals([2, 3, 1]));

        final sq = exp1.squeeze();
        expect(sq.shape, equals([2, 3]));
      });
    });
  });

  group(
    '4. Fancy Indexing: take_along_axis, put_along_axis, choose, select, where',
    () {
      test(
        'take_along_axis along axis 0 and 1 with positive and negative axes',
        () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [10, 20, 30, 40, 50, 60],
              [2, 3],
              DType.float64,
            );

            // Take along axis 1
            final idx1 = NDArray.fromList([2, 0, 1, 2], [2, 2], DType.int32);
            final res1 = take_along_axis(a, idx1, 1);
            expect(res1.shape, equals([2, 2]));
            expect(res1.toList(), equals([30.0, 10.0, 50.0, 60.0]));

            // Negative axis -1
            final resNeg1 = take_along_axis(a, idx1, -1);
            expect(resNeg1.toList(), equals(res1.toList()));

            // Take along axis 0
            final idx0 = NDArray.fromList([1, 0, 1], [1, 3], DType.int32);
            final res0 = take_along_axis(a, idx0, 0);
            expect(res0.shape, equals([1, 3]));
            expect(res0.toList(), equals([40.0, 20.0, 60.0]));
          });
        },
      );

      test('put_along_axis in-place and with out buffer', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [10, 20, 30, 40, 50, 60],
            [2, 3],
            DType.float64,
          );
          final idx = NDArray.fromList([1, 2, 0, 1], [2, 2], DType.int32);
          final vals = NDArray.fromList(
            [99.0, 88.0, 77.0, 66.0],
            [2, 2],
            DType.float64,
          );

          final out = NDArray.zeros([2, 3], DType.float64);
          put_along_axis(a, idx, vals, 1, out: out);
          expect(out.toList(), equals([10.0, 99.0, 88.0, 77.0, 66.0, 60.0]));
        });
      });

      test(
        'choose with raise, wrap, and clip modes across int & float dtypes',
        () {
          NDArray.scope(() {
            final choices = [
              NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64),
              NDArray.fromList([10.0, 11.0, 12.0], [3], DType.float64),
              NDArray.fromList([20.0, 21.0, 22.0], [3], DType.float64),
            ];

            // Standard raise mode
            final a1 = NDArray.fromList([0, 1, 2], [3], DType.int32);
            final res1 = choose(a1, choices, mode: ChooseMode.raise);
            expect(res1.toList(), equals([0.0, 11.0, 22.0]));

            // Wrap mode (wraps index 3 -> 0, -1 -> 2)
            final aWrap = NDArray.fromList([3, -1, 1], [3], DType.int32);
            final resWrap = choose(aWrap, choices, mode: ChooseMode.wrap);
            expect(resWrap.toList(), equals([0.0, 21.0, 12.0]));

            // Clip mode (clamps index 5 -> 2, -2 -> 0)
            final aClip = NDArray.fromList([-2, 5, 1], [3], DType.int32);
            final resClip = choose(aClip, choices, mode: ChooseMode.clip);
            expect(resClip.toList(), equals([0.0, 21.0, 12.0]));

            // Out-of-bounds error on raise mode
            final aBad = NDArray.fromList([0, 5, 1], [3], DType.int32);
            expect(
              () => choose(aBad, choices, mode: ChooseMode.raise),
              throwsRangeError,
            );
          });
        },
      );

      test('select with condition lists and default value', () {
        NDArray.scope(() {
          final c1 = NDArray.fromList(
            [true, false, false, false, false],
            [5],
            DType.boolean,
          );
          final c2 = NDArray.fromList(
            [false, false, false, true, true],
            [5],
            DType.boolean,
          );

          final ch1 = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          final ch2 = NDArray.fromList(
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [5],
            DType.float64,
          );

          final res = select([c1, c2], [ch1, ch2], defaultValue: -1.0);
          expect(res.toList(), equals([10.0, -1.0, -1.0, 400.0, 500.0]));
        });
      });

      test('where 3-arg multiplexer and 1-arg coordinate finder', () {
        NDArray.scope(() {
          final cond = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final y = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [4],
            DType.float64,
          );

          final res = where(cond, x, y) as NDArray;
          expect(res.toList(), equals([1.0, 20.0, 3.0, 40.0]));

          // 1-arg nonzero coordinates
          final coords = where(cond) as List<NDArray<AnySpec>>;
          expect(coords.length, equals(1));
          expect(coords[0].toList(), equals([0, 2]));
        });
      });
    },
  );

  group('5. Iteration & Scopes: NDIter, NDEnumerate, Scope Management', () {
    test('NDIter single array and multidimensional coordinates', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
        final iter = NDIter(a);
        final visitedCoords = <List<int>>[];
        final visitedIndices = <int>[];

        while (iter.moveNext()) {
          visitedCoords.add(List.from(iter.coords));
          visitedIndices.add(iter.index);
        }

        expect(
          visitedCoords,
          equals([
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
          ]),
        );
        expect(visitedIndices, equals([0, 1, 2, 3, 4, 5]));
      });
    });

    test('NDIter broadcast2 and broadcast3', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 1], DType.float64);
        final b = NDArray.zeros([1, 3], DType.float64);
        final iter2 = NDIter.broadcast2(a, b);
        var count = 0;
        while (iter2.moveNext()) {
          count++;
          expect(iter2.numArrays, equals(2));
        }
        expect(count, equals(6));

        final c = NDArray.zeros([2, 3], DType.float64);
        final iter3 = NDIter.broadcast3(a, b, c);
        var count3 = 0;
        while (iter3.moveNext()) {
          count3++;
          expect(iter3.numArrays, equals(3));
        }
        expect(count3, equals(6));
      });
    });

    test('NDEnumerate coordinates and values', () {
      NDArray.scope(() {
        final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
        final en = NDEnumerate<DTypeTag>(a);
        final collected = <int>[];
        while (en.moveNext()) {
          collected.add(en.value);
        }
        expect(collected, equals([10, 20, 30, 40]));
      });
    });

    test('NDArray.scope, returning, unmanaged, and leak tracking', () {
      NDArray<AnySpec>? escaped;
      NDArray.scope(() {
        final inside = NDArray.zeros([4], DType.float64);
        escaped = inside.detachToParentScope();
      });
      expect(escaped!.isDisposed, isFalse);
      expect(escaped!.size, equals(4));
      escaped!.dispose();
      expect(escaped!.isDisposed, isTrue);

      // NDArray.returning
      final returnedArr = NDArray.returning(() {
        return NDArray.ones([3], DType.float32);
      });
      expect(returnedArr.isDisposed, isFalse);
      expect(returnedArr.getCell([0]), equals(1.0));
      returnedArr.dispose();

      // NDArray.unmanaged
      final unmanagedArr = NDArray.unmanaged(() {
        return NDArray.zeros([2, 2], DType.int64);
      });
      expect(unmanagedArr.isDisposed, isFalse);
      unmanagedArr.dispose();
    });
  });

  group('6. Fast Fourier Transforms: 1D, 2D & N-D FFT/IFFT across DTypes', () {
    test(
      '1D fft & ifft across float32, float64, complex64, complex128, int32, bool',
      () {
        NDArray.scope(() {
          // Float64 input
          final f64 = NDArray.fromList(
            [1.0, 0.0, 0.0, 0.0],
            [4],
            DType.float64,
          );
          final resF64 = fft(f64);
          expect(resF64.dtype, equals(DType.complex128));
          for (var i = 0; i < 4; i++) {
            final c = resF64.getCell([i]);
            expect(c.real, closeTo(1.0, 1e-10));
            expect(c.imag, closeTo(0.0, 1e-10));
          }
          final invF64 = ifft(resF64);
          expect(invF64.getCell([0]).real, closeTo(1.0, 1e-10));
          expect(invF64.getCell([1]).real, closeTo(0.0, 1e-10));

          // Float32 input promotes to complex64
          final f32 = NDArray.fromList([2.0, 2.0], [2], DType.float32);
          final resF32 = fft(f32);
          expect(resF32.dtype, equals(DType.complex64));
          expect(resF32.getCell([0]).real, closeTo(4.0, 1e-5));
          expect(resF32.getCell([1]).real, closeTo(0.0, 1e-5));

          // Complex128 input
          final c128 = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            DType.complex128,
          );
          final resC128 = fft(c128);
          expect(resC128.getCell([0]).real, closeTo(4.0, 1e-10));
          expect(resC128.getCell([0]).imag, closeTo(6.0, 1e-10));

          // Int32 input
          final i32 = NDArray.fromList([1, 1, 1, 1], [4], DType.int32);
          final resI32 = fft(i32);
          expect(resI32.getCell([0]).real, closeTo(4.0, 1e-10));
          expect(resI32.getCell([0]).real, closeTo(4.0, 1e-10));

          // Boolean input
          final bArr = NDArray.fromList(
            [true, true, false, false],
            [4],
            DType.boolean,
          );
          final resBool = fft(bArr);
          expect(resBool.dtype, equals(DType.complex128));
          expect(resBool.getCell([0]).real, closeTo(2.0, 1e-10));
        });
      },
    );

    test('1D fft with target length n padding and truncation', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 1.0], [2], DType.float64);
        // Pad to length 4
        final padded = fft(a, n: 4);
        expect(padded.shape, equals([4]));
        // Truncate to length 1
        final trunc = fft(a, n: 1);
        expect(trunc.shape, equals([1]));
        expect(trunc.getCell([0]).real, closeTo(1.0, 1e-10));
      });
    });

    test('2D and N-D FFT: fft2, ifft2, fftn, ifftn with custom axes', () {
      NDArray.scope(() {
        final mat = NDArray.fromList(
          [1.0, 0.0, 0.0, 0.0],
          [2, 2],
          DType.float64,
        );
        final res2D = fft2(mat);
        expect(res2D.shape, equals([2, 2]));
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 2; j++) {
            expect(res2D.getCell([i, j]).real, closeTo(1.0, 1e-10));
          }
        }
        final inv2D = ifft2(res2D);
        expect(inv2D.getCell([0, 0]).real, closeTo(1.0, 1e-10));
        expect(inv2D.getCell([0, 1]).real, closeTo(0.0, 1e-10));

        // 3D FFT (fftn / ifftn)
        final tensor = NDArray.zeros([2, 2, 2], DType.float64);
        tensor[[0, 0, 0]] = 8.0;
        final res3D = fftn(tensor, axes: [0, 1, 2]);
        expect(res3D.shape, equals([2, 2, 2]));
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
              expect(res3D.getCell([i, j, k]).real, closeTo(8.0, 1e-10));
            }
          }
        }
        final inv3D = ifftn(res3D, axes: [0, 1, 2]);
        expect(inv3D.getCell([0, 0, 0]).real, closeTo(8.0, 1e-10));
      });
    });
  });

  group('7. Real Fast Fourier Transforms: rfft & irfft', () {
    test('1D rfft & irfft with even and odd output lengths', () {
      NDArray.scope(() {
        // Even length input N=4 -> rfft output size N/2 + 1 = 3
        final xEven = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final rEven = rfft(xEven);
        expect(rEven.shape, equals([3]));
        expect(rEven.dtype, equals(DType.complex128));
        // DC component
        expect(rEven.getCell([0]).real, closeTo(10.0, 1e-10));

        final restoredEven = irfft(rEven, n: 4);
        expect(restoredEven.shape, equals([4]));
        expect(restoredEven.getCell([0]), closeTo(1.0, 1e-10));
        expect(restoredEven.getCell([1]), closeTo(2.0, 1e-10));
        expect(restoredEven.getCell([2]), closeTo(3.0, 1e-10));
        expect(restoredEven.getCell([3]), closeTo(4.0, 1e-10));

        // Odd length input N=5 -> rfft output size (5~/2)+1 = 3
        final xOdd = NDArray.fromList(
          [1.0, 0.0, 1.0, 0.0, 1.0],
          [5],
          DType.float64,
        );
        final rOdd = rfft(xOdd);
        expect(rOdd.shape, equals([3]));
        final restoredOdd = irfft(rOdd, n: 5);
        expect(restoredOdd.shape, equals([5]));
        for (var i = 0; i < 5; i++) {
          expect(restoredOdd.getCell([i]), closeTo(xOdd.getCell([i]), 1e-10));
        }
      });
    });

    test('2D batch rfft along different axes', () {
      NDArray.scope(() {
        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 4],
          DType.float64,
        );

        final resAxis1 = rfft(mat, axis: 1);
        expect(resAxis1.shape, equals([2, 3]));

        final resAxis0 = rfft(mat, axis: 0);
        expect(resAxis0.shape, equals([2, 4]));
      });
    });
  });

  group(
    '8. FFT Frequencies & Spectral Shifts: fftfreq, rfftfreq, fftshift, ifftshift',
    () {
      test('fftfreq for even and odd lengths', () {
        NDArray.scope(() {
          final fEven = fftfreq(8, d: 1.0);
          expect(fEven.shape, equals([8]));
          expect(
            fEven.toList(),
            equals([0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]),
          );

          final fOdd = fftfreq(5, d: 1.0);
          expect(fOdd.shape, equals([5]));
          expect(fOdd.toList(), equals([0.0, 0.2, 0.4, -0.4, -0.2]));

          final rf = rfftfreq(5, d: 1.0);
          expect(rf.shape, equals([3]));
          expect(rf.toList(), equals([0.0, 0.2, 0.4]));

          // Spacing d = 0.5
          final fSpaced = fftfreq(4, d: 0.5);
          expect(fSpaced.toList(), equals([0.0, 0.5, -1.0, -0.5]));
        });
      });

      test(
        'fftshift and ifftshift 1D, 2D, and 3D round-trips with explicit axes',
        () {
          NDArray.scope(() {
            final a1Even = NDArray.fromList(
              [0, 1, 2, 3, 4, 5],
              [6],
              DType.int32,
            );
            final s1Even = fftshift(a1Even);
            expect(s1Even.toList(), equals([3, 4, 5, 0, 1, 2]));
            expect(ifftshift(s1Even).toList(), equals(a1Even.toList()));

            final a1Odd = NDArray.fromList([0, 1, 2, 3, 4], [5], DType.int32);
            final s1Odd = fftshift(a1Odd);
            expect(s1Odd.toList(), equals([3, 4, 0, 1, 2]));
            expect(ifftshift(s1Odd).toList(), equals(a1Odd.toList()));

            // 2D shift with axes
            final a2 = NDArray.fromList(
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              [3, 4],
              DType.int32,
            );
            final s2 = fftshift(a2, axes: [0, 1]);
            expect(ifftshift(s2, axes: [0, 1]).toList(), equals(a2.toList()));

            // Clear plan cache
            clearFFTPlanCache();
          });
        },
      );
    },
  );

  group('9. Standard Polynomials: polyval, polyfit, roots', () {
    test(
      'polyval Horner evaluation across float64, float32, complex128, complex64',
      () {
        NDArray.scope(() {
          // p(x) = 2x^2 - 4x + 5
          final c = NDArray.fromList([2.0, -4.0, 5.0], [3], DType.float64);
          final x = NDArray.fromList([0.0, 1.0, 2.0, 3.0], [4], DType.float64);
          final y = polyval(c, x);
          expect(y.toList(), equals([5.0, 3.0, 5.0, 11.0]));

          // Float32 evaluation
          final c32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
          final x32 = NDArray.fromList([3.0, 4.0], [2], DType.float32);
          final y32 = polyval(c32, x32);
          expect(y32.dtype, equals(DType.float32));
          expect(y32.getCell([0]), closeTo(5.0, 1e-5));
          expect(y32.getCell([1]), closeTo(6.0, 1e-5));

          // Complex128 evaluation
          final cComp = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 0.0)],
            [2],
            DType.complex128,
          );
          final xComp = NDArray.fromList(
            [Complex(0.0, 1.0)],
            [1],
            DType.complex128,
          );
          final yComp = polyval(cComp, xComp);
          expect(yComp.dtype, equals(DType.complex128));
          // (1+i)*i + 2 = (i - 1) + 2 = 1 + i
          expect(yComp.getCell([0]).real, closeTo(1.0, 1e-10));
          expect(yComp.getCell([0]).imag, closeTo(1.0, 1e-10));
        });
      },
    );

    test('polyfit least-squares polynomial fitting with full options', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.0, 1.0, 2.0, 3.0, 4.0],
          [5],
          DType.float64,
        );
        // y = 3x^2 - 2x + 1
        final y = NDArray.fromList(
          [1.0, 2.0, 9.0, 22.0, 41.0],
          [5],
          DType.float64,
        );
        final coeffs = polyfit(x, y, 2);
        expect(coeffs.shape, equals([3]));
        expect(coeffs.getCell([0]), closeTo(3.0, 1e-5));
        expect(coeffs.getCell([1]), closeTo(-2.0, 1e-5));
        expect(coeffs.getCell([2]), closeTo(1.0, 1e-5));

        // Linear fit with weights
        final w = NDArray.fromList(
          [1.0, 1.0, 2.0, 1.0, 1.0],
          [5],
          DType.float64,
        );
        final linCoeffs = polyfit(x, y, 1, w: w);
        expect(linCoeffs.shape, equals([2]));

        // Fit with explicit rcond
        final rcondCoeffs = polyfit(x, y, 2, rcond: 1e-10);
        expect(rcondCoeffs.getCell([0]), closeTo(3.0, 1e-5));

        // Degree 0 (constant fit)
        final constCoeffs = polyfit(x, y, 0);
        expect(constCoeffs.shape, equals([1]));
      });
    });

    test('roots of quadratic, cubic, linear and degenerate polynomials', () {
      NDArray.scope(() {
        // x^2 - 5x + 6 = 0 -> roots 3, 2
        final pQuad = NDArray.fromList([1.0, -5.0, 6.0], [3], DType.float64);
        final rQuad = roots(pQuad);
        expect(rQuad.shape, equals([2]));
        final rootVals = [
          rQuad.getCell([0]).real,
          rQuad.getCell([1]).real,
        ]..sort();
        expect(rootVals[0], closeTo(2.0, 1e-5));
        expect(rootVals[1], closeTo(3.0, 1e-5));

        // Linear: 2x - 8 = 0 -> root 4
        final pLin = NDArray.fromList([2.0, -8.0], [2], DType.float64);
        final rLin = roots(pLin);
        expect(rLin.shape, equals([1]));
        expect(rLin.getCell([0]).real, closeTo(4.0, 1e-10));

        // Leading zeros
        final pLeadZero = NDArray.fromList(
          [0.0, 0.0, 3.0, -9.0],
          [4],
          DType.float64,
        );
        final rLead = roots(pLeadZero);
        expect(rLead.shape, equals([1]));
        expect(rLead.getCell([0]).real, closeTo(3.0, 1e-10));

        // Degenerate (constant only) -> empty roots
        final pConst = NDArray.fromList([5.0], [1], DType.float64);
        final rConst = roots(pConst);
        expect(rConst.shape, equals([0]));
      });
    });
  });

  group('10. Orthogonal Polynomials: Chebyshev, Legendre, Hermite, Laguerre', () {
    test('chebval & chebroots', () {
      NDArray.scope(() {
        // T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2 - 1
        // Series: c = [1, 2, 3] -> 1*T_0 + 2*T_1 + 3*T_2 = 1 + 2x + 3(2x^2 - 1) = 6x^2 + 2x - 2
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 1.0, -1.0], [3], DType.float64);
        final y = chebval(x, c);
        expect(y.getCell([0]), closeTo(-2.0, 1e-6));
        expect(y.getCell([1]), closeTo(6.0, 1e-6));
        expect(y.getCell([2]), closeTo(2.0, 1e-6));

        // 2D array evaluation
        final x2D = NDArray.fromList(
          [0.0, 1.0, -1.0, 0.0],
          [2, 2],
          DType.float64,
        );
        final y2D = chebval(x2D, c);
        expect(y2D.shape, equals([2, 2]));
        expect(y2D.getCell([0, 0]), closeTo(-2.0, 1e-6));
        expect(y2D.getCell([0, 1]), closeTo(6.0, 1e-6));

        // chebroots: T_2(x) = 2x^2 - 1 = 0 -> x = +/- 1/sqrt(2) ~ +/- 0.707106
        final cT2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final rT2 = chebroots(cT2);
        expect(rT2.shape, equals([2]));
        final rVals = [
          rT2.getCell([0]).real,
          rT2.getCell([1]).real,
        ]..sort();
        expect(rVals[0], closeTo(-1.0 / math.sqrt(2.0), 1e-5));
        expect(rVals[1], closeTo(1.0 / math.sqrt(2.0), 1e-5));
      });
    });

    test('legval & legroots', () {
      NDArray.scope(() {
        // P_0(x) = 1, P_1(x) = x, P_2(x) = 0.5*(3x^2 - 1)
        final c = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 1.0], [2], DType.float64);
        final y = legval(x, c);
        expect(y.getCell([0]), closeTo(-0.5, 1e-6));
        expect(y.getCell([1]), closeTo(1.0, 1e-6));

        // legroots of P_2(x) = 0 -> x = +/- 1/sqrt(3) ~ +/- 0.57735
        final rLeg = legroots(c);
        expect(rLeg.shape, equals([2]));
        final rVals = [
          rLeg.getCell([0]).real,
          rLeg.getCell([1]).real,
        ]..sort();
        expect(rVals[0], closeTo(-1.0 / math.sqrt(3.0), 1e-5));
        expect(rVals[1], closeTo(1.0 / math.sqrt(3.0), 1e-5));
      });
    });

    test('hermval & hermroots', () {
      NDArray.scope(() {
        // H_0(x) = 1, H_1(x) = 2x, H_2(x) = 4x^2 - 2
        final c = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 1.0], [2], DType.float64);
        final y = hermval(x, c);
        expect(y.getCell([0]), closeTo(-2.0, 1e-6));
        expect(y.getCell([1]), closeTo(2.0, 1e-6));

        // hermroots of H_2(x) = 0 -> x = +/- 1/sqrt(2) ~ +/- 0.707106
        final rHerm = hermroots(c);
        expect(rHerm.shape, equals([2]));
        final rVals = [
          rHerm.getCell([0]).real,
          rHerm.getCell([1]).real,
        ]..sort();
        expect(rVals[0], closeTo(-1.0 / math.sqrt(2.0), 1e-5));
        expect(rVals[1], closeTo(1.0 / math.sqrt(2.0), 1e-5));
      });
    });

    test('lagval & lagroots', () {
      NDArray.scope(() {
        // L_0(x) = 1, L_1(x) = 1 - x, L_2(x) = 0.5*(x^2 - 4x + 2)
        final c = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 2.0], [2], DType.float64);
        final y = lagval(x, c);
        expect(y.getCell([0]), closeTo(1.0, 1e-6));
        expect(y.getCell([1]), closeTo(-1.0, 1e-6));

        // lagroots of L_2(x) = 0 -> roots 2 +/- sqrt(2) ~ 0.585786, 3.414213
        final rLag = lagroots(c);
        expect(rLag.shape, equals([2]));
        final rVals = [
          rLag.getCell([0]).real,
          rLag.getCell([1]).real,
        ]..sort();
        expect(rVals[0], closeTo(2.0 - math.sqrt(2.0), 1e-5));
        expect(rVals[1], closeTo(2.0 + math.sqrt(2.0), 1e-5));
      });
    });
  });

  group('11. 1D Scalar Root Finding: brentq, newton, secant, root_scalar', () {
    test('brentq root convergence, exact endpoints, and errors', () {
      // Normal root
      final res1 = brentq((x) => x * x * x - 8.0, 1.0, 3.0);
      expect(res1.converged, isTrue);
      expect(res1.root, closeTo(2.0, 1e-9));
      expect(res1.iterations, greaterThan(0));

      // Exact endpoint a
      final resA = brentq((x) => x - 1.0, 1.0, 4.0);
      expect(resA.converged, isTrue);
      expect(resA.root, equals(1.0));

      // Exact endpoint b
      final resB = brentq((x) => x - 4.0, 1.0, 4.0);
      expect(resB.converged, isTrue);
      expect(resB.root, equals(4.0));

      // Errors
      expect(() => brentq((x) => x * x + 1.0, 1.0, 3.0), throwsArgumentError);
      expect(() => brentq((x) => x, 0.0, 1.0, maxiter: 0), throwsArgumentError);
      expect(() => brentq((x) => x, 0.0, 1.0, xtol: -1.0), throwsArgumentError);
    });

    test('newton and secant iteration', () {
      // Newton with derivative
      final resN = newton((x) => x * x - 9.0, 2.0, fprime: (x) => 2.0 * x);
      expect(resN.converged, isTrue);
      expect(resN.root, closeTo(3.0, 1e-8));

      // Secant without derivative
      final resS = newton((x) => x * x - 9.0, 2.0);
      expect(resS.converged, isTrue);
      expect(resS.root, closeTo(3.0, 1e-8));

      // Derivative is zero error / failure
      final resZeroDeriv = newton(
        (x) => x * x + 1.0,
        0.0,
        fprime: (x) => 2.0 * x,
      );
      expect(resZeroDeriv.converged, isFalse);

      // Errors
      expect(() => newton((x) => x, 1.0, maxiter: 0), throwsArgumentError);
      expect(() => newton((x) => x, 1.0, tol: 0.0), throwsArgumentError);
    });

    test('root_scalar unified dispatcher', () {
      final resB = root_scalar(
        (x) => math.exp(x) - 3.0,
        method: RootMethod.brentq,
        bracketA: 0.0,
        bracketB: 2.0,
      );
      expect(resB.converged, isTrue);
      expect(resB.root, closeTo(math.log(3.0), 1e-8));

      final resN = root_scalar(
        (x) => math.exp(x) - 3.0,
        method: RootMethod.newton,
        x0: 1.0,
        fprime: (x) => math.exp(x),
      );
      expect(resN.converged, isTrue);
      expect(resN.root, closeTo(math.log(3.0), 1e-8));

      final resS = root_scalar(
        (x) => math.exp(x) - 3.0,
        method: RootMethod.secant,
        x0: 1.0,
      );
      expect(resS.converged, isTrue);
      expect(resS.root, closeTo(math.log(3.0), 1e-8));

      // Missing argument errors
      expect(
        () => root_scalar((x) => x, method: RootMethod.brentq),
        throwsArgumentError,
      );
      expect(
        () => root_scalar((x) => x, method: RootMethod.newton),
        throwsArgumentError,
      );
    });
  });

  group('12. Multivariate Optimization: nelder_mead, lbfgs, minimize', () {
    test('nelder_mead 2D optimization on parabolic bowl and Rosenbrock', () {
      NDArray.scope(() {
        // Parabolic bowl: f(x, y) = (x-3)^2 + (y+2)^2 + 5 -> minimum at (3, -2) with value 5
        double bowl(NDArray<Float64> x) {
          final x0 = x.getCell([0]);
          final x1 = x.getCell([1]);
          return (x0 - 3.0) * (x0 - 3.0) + (x1 + 2.0) * (x1 + 2.0) + 5.0;
        }

        final xInit = NDArray.fromList([0.0, 0.0], [2], DType.float64);
        final res = nelder_mead(bowl, xInit, adaptive: true);
        expect(res.success, isTrue);
        expect(res.x.getCell([0]), closeTo(3.0, 1e-3));
        expect(res.x.getCell([1]), closeTo(-2.0, 1e-3));
        expect(res.fun, closeTo(5.0, 1e-3));

        // CamelCase alias
        final resAlias = nelderMead(bowl, xInit);
        expect(resAlias.success, isTrue);
      });
    });

    test(
      'lbfgs optimization with analytical jacobian and numerical gradient',
      () {
        NDArray.scope(() {
          // Quadratic bowl f(x, y) = x^2 + 4y^2 -> minimum at (0, 0)
          double quad(NDArray<Float64> x) {
            final x0 = x.getCell([0]);
            final x1 = x.getCell([1]);
            return x0 * x0 + 4.0 * x1 * x1;
          }

          NDArray<Float64> quadJac(NDArray<Float64> x) {
            final x0 = x.getCell([0]);
            final x1 = x.getCell([1]);
            return NDArray.fromList([2.0 * x0, 8.0 * x1], [2], DType.float64);
          }

          final xInit = NDArray.fromList([5.0, -3.0], [2], DType.float64);

          // Analytical Jacobian
          final resJac = lbfgs(quad, xInit, jac: quadJac);
          expect(resJac.success, isTrue);
          expect(resJac.x.getCell([0]), closeTo(0.0, 1e-4));
          expect(resJac.x.getCell([1]), closeTo(0.0, 1e-4));
          expect(resJac.fun, closeTo(0.0, 1e-4));

          // Combined funAndGrad
          (double, NDArray<Float64>) funAndGrad(NDArray<Float64> x) {
            return (quad(x), quadJac(x));
          }

          final resCombined = lbfgs(quad, xInit, funAndGrad: funAndGrad);
          expect(resCombined.success, isTrue);

          // Numerical finite difference gradient
          final resNum = lbfgs(quad, xInit);
          expect(resNum.success, isTrue);
          expect(resNum.x.getCell([0]), closeTo(0.0, 1e-4));
        });
      },
    );

    test('minimize unified dispatcher and error checks', () {
      NDArray.scope(() {
        double sphere(NDArray<Float64> x) {
          var sum = 0.0;
          for (var i = 0; i < x.size; i++) {
            final v = x.getCellFlat(i);
            sum += (v - 1.0) * (v - 1.0);
          }
          return sum;
        }

        final x0 = NDArray.fromList([0.0, 0.0, 0.0], [3], DType.float64);

        final resNM = minimize(sphere, x0, method: MinimizeMethod.nelderMead);
        expect(resNM.success, isTrue);
        expect(resNM.x.getCell([0]), closeTo(1.0, 1e-3));

        final resLBFGS = minimize(sphere, x0, method: MinimizeMethod.lbfgs);
        expect(resLBFGS.success, isTrue);
        expect(resLBFGS.x.getCell([0]), closeTo(1.0, 1e-4));

        // Error conditions
        expect(
          () => nelder_mead(sphere, NDArray.zeros([2, 2], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => lbfgs(sphere, NDArray.zeros([2, 2], DType.float64)),
          throwsArgumentError,
        );
      });
    });
  });
}
