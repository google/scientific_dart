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

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

void main() {
  group(
    'Bug 1: NDArray.transpose, reshape, & ravel preserve offsetElements',
    () {
      test(
        'transpose, reshape, and ravel on views with non-zero offsetElements',
        () {
          final a = NDArray.fromList(List<int>.generate(16, (i) => i + 1), [
            4,
            4,
          ], DType.int32);
          try {
            final rev =
                a[[Slice(start: 3, stop: null, step: -1), Slice()]]
                    as NDArray<AnySpec>;
            expect(rev.offsetElements, equals(12));
            expect(
              rev.toList(),
              equals([13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]),
            );

            final revT = rev.transpose();
            expect(revT.offsetElements, equals(12));
            expect(
              revT.toList(),
              equals([13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 16, 12, 8, 4]),
            );

            final revTT = revT.transpose();
            expect(revTT.offsetElements, equals(12));
            expect(
              revTT.toList(),
              equals([13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]),
            );

            final sameShapeView = rev.reshape([4, 4]);
            expect(sameShapeView.offsetElements, equals(12));
            expect(
              sameShapeView.toList(),
              equals([13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]),
            );

            final reshapedRev = rev.reshape([2, 8]);
            try {
              expect(
                reshapedRev.toList(),
                equals([13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]),
              );
            } finally {
              reshapedRev.dispose();
            }

            final flipped1d = a.ravel().slice([Slice.all(step: -1)]);
            expect(flipped1d.offsetElements, equals(15));
            final transposed1d = flipped1d.transpose();
            expect(transposed1d.offsetElements, equals(15));
            expect(
              transposed1d.toList(),
              equals(List<int>.generate(16, (i) => 16 - i)),
            );
          } finally {
            a.dispose();
          }
        },
      );
    },
  );

  group('Bug 2: Empty arrays with 0 in shape isContiguous & copy operations', () {
    test(
      '0-dimension arrays and empty slices report isContiguous true and copy/flatten safely',
      () {
        for (final shape in [
          [0],
          [0, 5],
          [5, 0],
          [3, 0, 2],
        ]) {
          final empty = NDArray.create(shape, DType.float64);
          try {
            expect(empty.size, equals(0));
            expect(empty.isContiguous, isTrue);

            final c1 = empty.copy();
            final f1 = empty.flatten();
            final ccDest = NDArray.create(shape, DType.float64);
            empty.copyToContiguous(ccDest);
            try {
              expect(c1.shape, equals(shape));
              expect(f1.shape, equals([0]));
              expect(ccDest.shape, equals(shape));
            } finally {
              c1.dispose();
              f1.dispose();
              ccDest.dispose();
            }
          } finally {
            empty.dispose();
          }
        }

        final parent = NDArray.zeros([5, 2], DType.float64);
        try {
          final emptySlice = parent.slice([
            Slice.all(),
            Slice(start: 0, stop: 0),
          ]);
          expect(emptySlice.shape, equals([5, 0]));
          expect(emptySlice.isContiguous, isTrue);
          final dest = NDArray.create([5, 0], DType.float64);
          try {
            emptySlice.copyToContiguous(dest);
            expect(dest.shape, equals([5, 0]));
          } finally {
            dest.dispose();
          }
        } finally {
          parent.dispose();
        }
      },
    );
  });

  group(
    'Bug 3: Bounds checking in NDArray.view & custom/negative strides in create/fromPointer',
    () {
      test(
        'NDArray.create with negative and custom strides allocates sufficient buffer and disposes safely',
        () {
          final negArr = NDArray.create([4], DType.int32, strides: [-1]);
          try {
            expect(negArr.offsetElements, equals(3));
            negArr[0] = 10;
            negArr[1] = 20;
            negArr[2] = 30;
            negArr[3] = 40;
            expect(negArr.toList(), equals([10, 20, 30, 40]));
          } finally {
            negArr.dispose();
          }

          final gapArr = NDArray.create([3], DType.int32, strides: [2]);
          try {
            expect(gapArr.offsetElements, equals(0));
            expect(gapArr.data.length, equals(5));
            gapArr[0] = 11;
            gapArr[1] = 22;
            gapArr[2] = 33;
            expect(gapArr.toList(), equals([11, 22, 33]));
          } finally {
            gapArr.dispose();
          }
        },
      );

      test(
        'NDArray.fromPointer with negative strides computes offsetElements and disposes via nativeFinalizer',
        () {
          final rawPtr = malloc.allocate<ffi.Int32>(
            4 * ffi.sizeOf<ffi.Int32>(),
          );
          rawPtr[0] = 100;
          rawPtr[1] = 200;
          rawPtr[2] = 300;
          rawPtr[3] = 400;
          final extNeg = NDArray<Int32>.fromPointer(
            rawPtr.cast(),
            [4],
            DType.int32,
            strides: [-1],
            nativeFinalizer: _isWasm ? null : malloc.nativeFree,
          );
          expect(extNeg.offsetElements, equals(3));
          expect(extNeg.toList(), equals([400, 300, 200, 100]));
          extNeg.dispose();
          if (_isWasm) {
            malloc.free(rawPtr);
          }
        },
      );

      test(
        'NDArray.view throws RangeError when offset or strides exceed root allocation',
        () {
          final base = NDArray.zeros([4], DType.int32);
          try {
            expect(
              () => NDArray.view(
                base,
                shape: [5],
                strides: [1],
                offsetElements: 0,
              ),
              throwsRangeError,
            );
            expect(
              () => NDArray.view(
                base,
                shape: [2],
                strides: [1],
                offsetElements: 3,
              ),
              throwsRangeError,
            );
            expect(
              () => NDArray.view(
                base,
                shape: [3],
                strides: [-1],
                offsetElements: 1,
              ),
              throwsRangeError,
            );
          } finally {
            base.dispose();
          }
        },
      );
    },
  );

  group(
    'Bug 4: Strided casting (s_cast_generic) and astype for float16 and bfloat16',
    () {
      test(
        'contiguous and non-contiguous float16 and bfloat16 cast accurately to/from other DTypes',
        () {
          final f64 = NDArray.fromList(
            [1.5, -2.25, 4.0, 8.5],
            [2, 2],
            DType.float64,
          );
          try {
            final f64T = f64.transpose();
            expect(f64T.isContiguous, isFalse);

            final f16 = f64T.astype<Float16>(DType.float16);
            final bf16 = f64T.astype<BFloat16>(DType.bfloat16);
            try {
              expect(f16.toList(), equals([1.5, 4.0, -2.25, 8.5]));
              expect(bf16.toList(), equals([1.5, 4.0, -2.25, 8.5]));

              final f16T = f16.transpose();
              final bf16T = bf16.transpose();
              expect(f16T.isContiguous, isFalse);
              expect(bf16T.isContiguous, isFalse);

              final backFromF16 = f16T.astype<Float64>(DType.float64);
              final backFromBf16 = bf16T.astype<Int32>(DType.int32);
              final crossCast = f16T.astype<BFloat16>(DType.bfloat16);
              try {
                expect(backFromF16.toList(), equals([1.5, -2.25, 4.0, 8.5]));
                expect(backFromBf16.toList(), equals([1, -2, 4, 8]));
                expect(crossCast.toList(), equals([1.5, -2.25, 4.0, 8.5]));
              } finally {
                backFromF16.dispose();
                backFromBf16.dispose();
                crossCast.dispose();
              }
            } finally {
              f16.dispose();
              bf16.dispose();
            }
          } finally {
            f64.dispose();
          }
        },
      );
    },
  );

  group('Bug 5: N-D boolean masking & operator [] integer NDArray indexing', () {
    test(
      'N-D boolean masking on contiguous and transposed arrays and assignment with NDArray<AnySpec>',
      () {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        try {
          final mask = a > 3.0;
          try {
            final masked = a[mask] as NDArray<AnySpec>;
            try {
              expect(masked.shape, equals([3]));
              expect(masked.toList(), equals([4.0, 5.0, 6.0]));
            } finally {
              masked.dispose();
            }

            final aT = a.transpose();
            final maskT = mask.transpose();
            final maskedT = aT.applyMask(maskT);
            try {
              expect(maskedT.toList(), equals([4.0, 5.0, 6.0]));
            } finally {
              maskedT.dispose();
            }

            final scaled = a[mask] * 10.0;
            try {
              a[mask] = scaled;
              expect(a.toList(), equals([1.0, 2.0, 3.0, 40.0, 50.0, 60.0]));
            } finally {
              if (scaled is NDArray) scaled.dispose();
            }
          } finally {
            mask.dispose();
          }
        } finally {
          a.dispose();
        }
      },
    );

    test(
      'integer NDArray indexing and assignment works for same-shape and 2D+ selectors with broadcasting',
      () {
        final arr = NDArray.fromList([10, 20, 30], [3], DType.int32);
        final idxSameShape = NDArray.fromList([2, 0, 1], [3], DType.int32);
        final idx2D = NDArray.fromList([2, 0, 1, 2], [2, 2], DType.int32);
        try {
          final res1 = arr[idxSameShape] as NDArray<AnySpec>;
          final res2 = arr[idx2D] as NDArray<AnySpec>;
          try {
            expect(res1.shape, equals([3]));
            expect(res1.toList(), equals([30, 10, 20]));

            expect(res2.shape, equals([2, 2]));
            expect(res2.toList(), equals([30, 10, 20, 30]));
          } finally {
            res1.dispose();
            res2.dispose();
            expect(res2.isDisposed, isTrue);
          }

          final valsSame = NDArray.fromList([100, 200, 300], [3], DType.int32);
          try {
            arr[idxSameShape] = valsSame;
            expect(arr.toList(), equals([200, 300, 100]));
          } finally {
            valsSame.dispose();
          }

          arr[idx2D] = -5;
          expect(arr.toList(), equals([-5, -5, -5]));

          // 2D target array indexed by 2D integer NDArray with broadcast 1D value
          final mat = NDArray.zeros([4, 3], DType.int32);
          final rowIdx2D = NDArray.fromList([0, 2, 1, 3], [2, 2], DType.int32);
          final rowVal = NDArray.fromList([7, 8, 9], [3], DType.int32);
          try {
            mat[rowIdx2D] = rowVal;
            expect(mat.toList(), equals([7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9]));
          } finally {
            mat.dispose();
            rowIdx2D.dispose();
            rowVal.dispose();
          }
        } finally {
          arr.dispose();
          idxSameShape.dispose();
          idx2D.dispose();
        }
      },
    );
  });

  group('Bug 6: Negative index assignment & out-of-range scalar comparisons', () {
    test(
      'operator []=, getCell, setCell, and setIndicesScalar support negative integer indices',
      () {
        final a = NDArray.fromList([10, 20, 30], [3], DType.int32);
        final b = NDArray.zeros([2, 2], DType.int32);
        try {
          a[-1] = 99;
          expect(a.toList(), equals([10, 20, 99]));

          b[[-1, -1]] = 42;
          expect(b[[-1, -1]], equals(42));
          expect(b.getCell([-1, -1]), equals(42));
          b.setCell([-2, -1], 17);
          expect(b.toList(), equals([0, 17, 0, 42]));

          final negIdx = NDArray.fromList([-1, -3], [2], DType.int32);
          try {
            a.setIndicesScalar(negIdx, -1);
            expect(a.toList(), equals([-1, 20, -1]));
          } finally {
            negIdx.dispose();
          }
        } finally {
          a.dispose();
          b.dispose();
        }
      },
    );

    test(
      'out-of-range scalar comparisons do not modulo-wrap into target dtype and mixed signed/unsigned compare accurately',
      () {
        final u8 = NDArray.fromList([44], [1], DType.uint8);
        final u64 = NDArray.fromList([10], [1], DType.uint64);
        final i64 = NDArray.fromList([-1], [1], DType.int64);
        try {
          // With saturating scalar casts (P7): 300 saturates to 255 in uint8, -1 saturates to 0 in uint8/uint64
          final eq300 = u8.eq(300);
          final ltNeg1 = u8 < -1;
          final gtNeg1 = u8 > -1;
          final lt300 = u8 < 300;
          final u64LtNeg1 = u64 < -1;
          final u64EqNeg1 = u64.eq(-1);
          expect(() => less(i64.asAnySpec, u64.asAnySpec), throwsArgumentError);
          expect(
            () => greater(u64.asAnySpec, i64.asAnySpec),
            throwsArgumentError,
          );
          try {
            expect(eq300.toList(), equals([false]));
            expect(ltNeg1.toList(), equals([false]));
            expect(gtNeg1.toList(), equals([true]));
            expect(lt300.toList(), equals([true]));
            expect(u64LtNeg1.toList(), equals([false]));
            expect(u64EqNeg1.toList(), equals([false]));
          } finally {
            eq300.dispose();
            ltNeg1.dispose();
            gtNeg1.dispose();
            lt300.dispose();
            u64LtNeg1.dispose();
            u64EqNeg1.dispose();
          }
        } finally {
          u8.dispose();
          u64.dispose();
          i64.dispose();
        }
      },
    );
  });

  group(
    'Bug 7: In-place overlapping slice copy uses memmove / temporary buffer',
    () {
      test(
        'overlapping contiguous forward/backward shift and reversed slice assignment do not corrupt data',
        () {
          final a = NDArray.fromList([1, 2, 3, 4, 5], [5], DType.int32);
          final b = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);
          final c = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          try {
            final srcA = a[Slice(start: 0, stop: 4)] as NDArray<AnySpec>;
            final dstA = a[Slice(start: 1, stop: 5)] as NDArray<AnySpec>;
            srcA.copy(out: dstA);
            expect(a.toList(), equals([1, 1, 2, 3, 4]));

            b[Slice(start: 0, stop: 4)] = b[Slice(start: 1, stop: 5)];
            expect(b.toList(), equals([20, 30, 40, 50, 50]));

            c[Slice.all()] = c.slice([Slice(step: -1)]);
            expect(c.toList(), equals([4, 3, 2, 1]));
          } finally {
            a.dispose();
            b.dispose();
            c.dispose();
          }
        },
      );

      test(
        'overlapping non-contiguous strided 2D slice copy and setByMask/setIndices aliasing preserve elements',
        () {
          final a = NDArray.fromList(List<int>.generate(9, (i) => i + 1), [
            3,
            3,
          ], DType.int32);
          final m = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final mask = NDArray.fromList(
            [true, true, true, true],
            [4],
            DType.boolean,
          );
          final idx = NDArray.fromList([3, 2, 1, 0], [4], DType.int32);
          try {
            final src =
                a[[Slice(start: 0, stop: 2), Slice(start: 0, stop: 2)]]
                    as NDArray<AnySpec>;
            final dst =
                a[[Slice(start: 1, stop: 3), Slice(start: 1, stop: 3)]]
                    as NDArray<AnySpec>;
            src.copy(out: dst);
            expect(a.toList(), equals([1, 2, 3, 4, 1, 2, 7, 4, 5]));

            // Aliasing in setByMask
            m.setByMask(mask, m.slice([Slice(step: -1)]));
            expect(m.toList(), equals([4, 3, 2, 1]));

            // Aliasing in setIndices
            m.setIndices(idx, m);
            expect(m.toList(), equals([1, 2, 3, 4]));
          } finally {
            a.dispose();
            m.dispose();
            mask.dispose();
            idx.dispose();
          }
        },
      );
    },
  );

  group(
    'Bug 8: equals() all 15 DTypes, IEEE 754 float equality, & contentHashCode negative ints',
    () {
      test(
        'equals() works for float16, bfloat16, int8, uint16, uint32, uint64 on non-contiguous views',
        () {
          for (final dt in [
            DType.float16,
            DType.bfloat16,
            DType.int8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
          ]) {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], dt);
            final b = NDArray.fromList([1, 3, 2, 4], [2, 2], dt);
            try {
              final bT = b.transpose();
              expect(
                a.equals(bT),
                isTrue,
                reason: 'Failed a.equals(b.T) for $dt',
              );
              expect(
                bT.equals(b.transpose()),
                isTrue,
                reason: 'Failed b.T.equals(b.T) for $dt',
              );
            } finally {
              a.dispose();
              b.dispose();
            }
          }
        },
      );

      test(
        'equals() adheres to IEEE 754 for +0.0 == -0.0 and NaN != NaN across float DTypes',
        () {
          for (final dt in [
            DType.float64,
            DType.float32,
            DType.float16,
            DType.bfloat16,
          ]) {
            final posZero = NDArray.fromList([0.0], [1], dt);
            final negZero = NDArray.fromList([-0.0], [1], dt);
            final nanArr1 = NDArray.fromList([double.nan], [1], dt);
            final nanArr2 = NDArray.fromList([double.nan], [1], dt);
            try {
              expect(
                posZero.equals(negZero),
                isTrue,
                reason: '+0.0 == -0.0 failed for $dt',
              );
              expect(
                posZero.contentHashCode,
                equals(negZero.contentHashCode),
                reason: 'contentHashCode for -0.0 failed for $dt',
              );
              expect(
                nanArr1.equals(nanArr2),
                isFalse,
                reason: 'NaN == NaN must be false for $dt',
              );
            } finally {
              posZero.dispose();
              negZero.dispose();
              nanArr1.dispose();
              nanArr2.dispose();
            }
          }
        },
      );

      test(
        'contentHashCode does not collide to 0 or NaN canonical bits for negative int64/int32 arrays',
        () {
          final n1 = NDArray.fromList([-1, -2], [2], DType.int64);
          final n2 = NDArray.fromList([-100, -200], [2], DType.int64);
          final i64Min = NDArray.fromList(
            [-9223372036854775808],
            [1],
            DType.int64,
          );
          final i64Zero = NDArray.fromList([0], [1], DType.int64);
          final n3 = NDArray.fromList([-1, -2], [2], DType.int32);
          final n4 = NDArray.fromList([-100, -200], [2], DType.int32);
          final i32Min = NDArray.fromList([-2147483648], [1], DType.int32);
          final i32Zero = NDArray.fromList([0], [1], DType.int32);
          try {
            expect(n1.contentHashCode, isNot(equals(0)));
            expect(n1.contentHashCode, isNot(equals(n2.contentHashCode)));
            expect(
              i64Min.contentHashCode,
              isNot(equals(i64Zero.contentHashCode)),
            );
            expect(n3.contentHashCode, isNot(equals(0)));
            expect(n3.contentHashCode, isNot(equals(n4.contentHashCode)));
            expect(
              i32Min.contentHashCode,
              isNot(equals(i32Zero.contentHashCode)),
            );
          } finally {
            n1.dispose();
            n2.dispose();
            i64Min.dispose();
            i64Zero.dispose();
            n3.dispose();
            n4.dispose();
            i32Min.dispose();
            i32Zero.dispose();
          }
        },
      );
    },
  );

  group(
    'Bug 9: Set operations support all DTypes, float16/bfloat16 sort order, mixed DTypes, and safe disposal',
    () {
      test(
        'unique sorts negative float16/bfloat16 values correctly and deduplicates NaN',
        () {
          for (final dt in [DType.float16, DType.bfloat16]) {
            final arr = NDArray.fromList(
              [1.5, -0.5, -2.0, 0.0, -2.0, double.nan, double.nan],
              [7],
              dt,
            );
            try {
              final u = unique(arr);
              final uWithCounts = uniqueWithCounts(arr);
              try {
                expect(u.size, equals(5));
                final list = u.toList();
                expect(list.sublist(0, 4), equals([-2.0, -0.5, 0.0, 1.5]));
                expect(list[4].isNaN, isTrue);

                expect(uWithCounts.values.size, equals(5));
                expect(uWithCounts.counts.toList(), equals([2, 1, 1, 1, 2]));
              } finally {
                u.dispose();
                uWithCounts.values.dispose();
                uWithCounts.counts.dispose();
              }
            } finally {
              arr.dispose();
            }
          }
        },
      );

      test(
        'intersect1d, union1d, setdiff1d, setxor1d, isin handle mixed DTypes, uint16/int8, and do not dispose equal inputs',
        () {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int8);
          final b = NDArray.fromList([2, 3, 4], [3], DType.int16);
          final equalA = NDArray.fromList([1, 2, 3], [3], DType.int8);
          try {
            final inter = intersect1d(a.asAnySpec, b.asAnySpec);
            final uni = union1d(a.asAnySpec, b.asAnySpec);
            final diff = setdiff1d(a.asAnySpec, b.asAnySpec);
            final xor = setxor1d(a.asAnySpec, b.asAnySpec);
            final inMask = isin<DTypeTag>(a, b);
            final sameInter = intersect1d(a, equalA);
            try {
              expect(inter.toList(), equals([2, 3]));
              expect(uni.toList(), equals([1, 2, 3, 4]));
              expect(diff.toList(), equals([1]));
              expect(xor.toList(), equals([1, 4]));
              expect(inMask.toList(), equals([false, true, true]));
              expect(sameInter.toList(), equals([1, 2, 3]));
              expect(a.isDisposed, isFalse);
              expect(equalA.isDisposed, isFalse);
            } finally {
              inter.dispose();
              uni.dispose();
              diff.dispose();
              xor.dispose();
              inMask.dispose();
              sameInter.dispose();
            }
          } finally {
            a.dispose();
            b.dispose();
            equalA.dispose();
          }
        },
      );
    },
  );

  group('Bug 10: NPY v2.0 4-byte header length loader support in load() and loadz()', () {
    test(
      'load() and loadz() parse NPY v2.0 files including >64KB headers where u32Len & 0xFFFF is small',
      () {
        final tempDir = Directory.systemTemp.createTempSync('npy_v2_test_');
        try {
          // Construct an NPY v2.0 header with u32Len = 65540 (where u32Len & 0xFFFF == 4, so rawHeaderLen = 14)
          const dictStr =
              "{'descr': '<i4', 'fortran_order': False, 'shape': (3,), }";
          const targetHeaderLen =
              65540; // 12 + 65540 = 65552, divisible by 16; let's pad dictStr at END of 65540 bytes!
          final padCount = targetHeaderLen - dictStr.length - 1;
          // Put spaces BEFORE dictStr so that the first 14 bytes of the header contain only spaces!
          final fullHeaderAscii = '${' ' * padCount}$dictStr\n';

          final builder = BytesBuilder();
          builder.add([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]); // NUMPY
          builder.add([0x02, 0x00]); // v2.0
          final lenData = ByteData(4)
            ..setUint32(0, targetHeaderLen, Endian.little);
          builder.add(lenData.buffer.asUint8List());
          builder.add(ascii.encode(fullHeaderAscii));

          final payload = Int32List.fromList([100, 200, 300]);
          builder.add(payload.buffer.asUint8List());
          final npyBytes = builder.toBytes();

          final npyPath = '${tempDir.path}/test_v2_large.npy';
          File(npyPath).writeAsBytesSync(npyBytes);

          // 1. Test load()
          final loaded = load(npyPath);
          try {
            expect(loaded.dtype, equals(DType.int32));
            expect(loaded.shape, equals([3]));
            expect(loaded.toList(), equals([100, 200, 300]));
          } finally {
            loaded.dispose();
          }

          // 2. Test loadz() with >64KB header NPY v2.0 entry where dictStr is at offset > 65000
          final archive = Archive();
          archive.addFile(
            ArchiveFile('arr_large.npy', npyBytes.length, npyBytes),
          );
          final zipBytes = ZipEncoder().encode(archive)!;
          final npzPath = '${tempDir.path}/test_v2_large.npz';
          File(npzPath).writeAsBytesSync(zipBytes);

          final loadedMap = loadz(npzPath);
          try {
            expect(loadedMap.containsKey('arr_large'), isTrue);
            final arr = loadedMap['arr_large']!;
            expect(arr.dtype, equals(DType.int32));
            expect(arr.shape, equals([3]));
            expect(arr.toList(), equals([100, 200, 300]));
          } finally {
            for (final a in loadedMap.values) {
              a.dispose();
            }
          }
        } finally {
          tempDir.deleteSync(recursive: true);
        }
      },
      skip: const bool.fromEnvironment('dart.tool.dart2wasm')
          ? 'Uses dart:io File.writeAsBytesSync'
          : false,
    );
  });
}
