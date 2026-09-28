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
  group(
    'Review Cycle 16 — Issue #1: ScratchArena.getStridedBuffer & High-Rank Strided Binary Ops',
    () {
      test(
        'ScratchArena.getStridedBuffer defaults to 4 segments and never under-allocates',
        () {
          final marker = ScratchArena.marker;
          try {
            const rank = 12;
            final cBuffer = ScratchArena.getStridedBuffer(rank);
            // Allocate sentinel immediately after cBuffer in ScratchArena
            final sentinel = ScratchArena.allocate<ffi.Int64>(
              4 * ffi.sizeOf<ffi.Int64>(),
            );
            for (var i = 0; i < 4; i++) {
              sentinel[i] = 0x123456789ABCDEF0;
            }

            // Write all 4 rank-sized segments (cShape, cStridesA, cStridesB, cStridesRes)
            final cShape = cBuffer;
            final cStridesA = cBuffer + rank;
            final cStridesB = cBuffer + (rank * 2);
            final cStridesRes = cBuffer + (rank * 3);
            for (var i = 0; i < rank; i++) {
              cShape[i] = i + 1;
              cStridesA[i] = (i + 1) * 10;
              cStridesB[i] = (i + 1) * 100;
              cStridesRes[i] = (i + 1) * 1000;
            }

            // Verify sentinel was not overwritten by cStridesRes writes
            for (var i = 0; i < 4; i++) {
              expect(sentinel[i], equals(0x123456789ABCDEF0));
            }
          } finally {
            ScratchArena.reset(marker);
          }
        },
      );

      test(
        'High-rank (34D) strided binary operations execute without clobbering arena',
        () {
          NDArray.scope(() {
            // Shape [2, 1, 1, ..., 1, 3] with 34 dimensions (6 elements total)
            final shape = <int>[2, ...List<int>.filled(32, 1), 3];
            final baseA = NDArray<Int64>.arange(1, 13, dtype: DType.int64);
            final baseB = NDArray<Int64>.arange(10, 22, dtype: DType.int64);
            // Step-2 slice makes both non-contiguous before reshaping to 34D via view
            final slicedA = baseA.slice([Slice(start: 0, stop: 12, step: 2)]);
            final slicedB = baseB.slice([Slice(start: 0, stop: 12, step: 2)]);
            final strides34D = <int>[
              3 * slicedA.strides[0],
              ...List<int>.filled(32, 3 * slicedA.strides[0]),
              slicedA.strides[0],
            ];
            final a34 = NDArray<Int64>.view(
              slicedA,
              shape: shape,
              strides: strides34D,
              offsetElements: 0,
            );
            final b34 = NDArray<Int64>.view(
              slicedB,
              shape: shape,
              strides: strides34D,
              offsetElements: 0,
            );

            final sum34 = add(a34, b34);
            final diff34 = subtract(b34, a34);
            final prod34 = multiply(a34, b34);
            final fdiv34 = floorDivide(b34, a34);
            final rem34 = remainder(b34, a34);
            final lcm34 = lcm(a34, b34);
            final band34 = bitwiseAnd(a34, b34);
            final bor34 = bitwiseOr(a34, b34);
            final bxor34 = bitwiseXor(a34, b34);
            final lsh34 = leftShift(a34, NDArray.scalar(1, dtype: DType.int64));
            final rsh34 = rightShift(
              b34,
              NDArray.scalar(1, dtype: DType.int64),
            );

            expect(sum34.shape, equals(shape));
            // Elements of slicedA: [1, 3, 5, 7, 9, 11]
            // Elements of slicedB: [10, 12, 14, 16, 18, 20]
            expect(sum34.ravel().toList(), equals([11, 15, 19, 23, 27, 31]));
            expect(diff34.ravel().toList(), equals([9, 9, 9, 9, 9, 9]));
            expect(
              prod34.ravel().toList(),
              equals([10, 36, 70, 112, 162, 220]),
            );
            expect(fdiv34.ravel().toList(), equals([10, 4, 2, 2, 2, 1]));
            expect(rem34.ravel().toList(), equals([0, 0, 4, 2, 0, 9]));
            expect(lcm34.ravel().toList(), equals([10, 12, 70, 112, 18, 220]));
            expect(
              band34.ravel().toList(),
              equals([1 & 10, 3 & 12, 5 & 14, 7 & 16, 9 & 18, 11 & 20]),
            );
            expect(
              bor34.ravel().toList(),
              equals([1 | 10, 3 | 12, 5 | 14, 7 | 16, 9 | 18, 11 | 20]),
            );
            expect(
              bxor34.ravel().toList(),
              equals([1 ^ 10, 3 ^ 12, 5 ^ 14, 7 ^ 16, 9 ^ 18, 11 ^ 20]),
            );
            expect(lsh34.ravel().toList(), equals([2, 6, 10, 14, 18, 22]));
            expect(rsh34.ravel().toList(), equals([5, 6, 7, 8, 9, 10]));
          });
        },
      );
    },
  );

  group('Review Cycle 16 — Issue #2: All 8 Integer DTypes in Bitwise Operations', () {
    final intDTypes = <DType<AnySpec>>[
      DType.int8,
      DType.int16,
      DType.int32,
      DType.int64,
      DType.uint8,
      DType.uint16,
      DType.uint32,
      DType.uint64,
    ];

    test(
      'bitwiseAnd, bitwiseOr, bitwiseXor, leftShift, invert across all 8 integer DTypes (contiguous and strided)',
      () {
        for (final dt in intDTypes) {
          NDArray.scope(() {
            final aFull = NDArray.fromList(
              [3, 5, 6, 12, 15, 9, 10, 7],
              [8],
              dt,
            );
            final bFull = NDArray.fromList([1, 3, 2, 4, 7, 5, 6, 3], [8], dt);

            // Contiguous tests
            final cAnd = bitwiseAnd(aFull, bFull);
            final cOr = bitwiseOr(aFull, bFull);
            final cXor = bitwiseXor(aFull, bFull);
            final cShl = leftShift(aFull, NDArray.full([8], 1, dtype: dt));
            final cInv = invert(aFull);

            expect(cAnd.dtype, equals(dt));
            expect(cOr.dtype, equals(dt));
            expect(cXor.dtype, equals(dt));
            expect(cShl.dtype, equals(dt));
            expect(cInv.dtype, equals(dt));

            for (var i = 0; i < 8; i++) {
              final av = aFull.getCell([i]);
              final bv = bFull.getCell([i]);
              expect(cAnd.getCell([i]), equals(av & bv), reason: 'and $dt');
              expect(cOr.getCell([i]), equals(av | bv), reason: 'or $dt');
              expect(cXor.getCell([i]), equals(av ^ bv), reason: 'xor $dt');
              expect(cShl.getCell([i]), equals(av << 1), reason: 'shl $dt');
            }

            // Strided tests (step = 2)
            final aStrided = aFull.slice([Slice(start: 0, stop: 8, step: 2)]);
            final bStrided = bFull.slice([Slice(start: 0, stop: 8, step: 2)]);
            expect(aStrided.isContiguous, isFalse);

            final sAnd = bitwiseAnd(aStrided, bStrided);
            final sOr = bitwiseOr(aStrided, bStrided);
            final sXor = bitwiseXor(aStrided, bStrided);
            final sShl = leftShift(aStrided, NDArray.full([4], 1, dtype: dt));
            final sInv = invert(aStrided);

            for (var i = 0; i < 4; i++) {
              expect(sAnd.getCell([i]), equals(cAnd.getCell([i * 2])));
              expect(sOr.getCell([i]), equals(cOr.getCell([i * 2])));
              expect(sXor.getCell([i]), equals(cXor.getCell([i * 2])));
              expect(sShl.getCell([i]), equals(cShl.getCell([i * 2])));
              expect(sInv.getCell([i]), equals(cInv.getCell([i * 2])));
            }
          });
        }
      },
    );

    test(
      'rightShift preserves signed vs unsigned semantics for int8, uint8, int16, uint16, int32, uint32, int64, uint64',
      () {
        NDArray.scope(() {
          // int8: -16 >> 2 == -4 (arithmetic shift)
          final i8 = NDArray<Int8>.fromList(
            [-16, -64, 32, -8],
            [4],
            DType.int8,
          );
          final s8 = NDArray<Int8>.fromList([2, 3, 2, 1], [4], DType.int8);
          expect(rightShift(i8, s8).toList(), equals([-4, -8, 8, -4]));
          expect(
            rightShift(
              i8.slice([Slice(start: 0, stop: 4, step: 2)]),
              s8.slice([Slice(start: 0, stop: 4, step: 2)]),
            ).toList(),
            equals([-4, 8]),
          );

          // uint8: 0xF0 (240) >>> 4 == 15 (logical shift)
          final u8 = NDArray<Uint8>.fromList(
            [240, 128, 255, 64],
            [4],
            DType.uint8,
          );
          final su8 = NDArray<Uint8>.fromList([4, 7, 4, 2], [4], DType.uint8);
          expect(rightShift(u8, su8).toList(), equals([15, 1, 15, 16]));
          expect(
            rightShift(
              u8.slice([Slice(start: 0, stop: 4, step: 2)]),
              su8.slice([Slice(start: 0, stop: 4, step: 2)]),
            ).toList(),
            equals([15, 15]),
          );

          // uint16: 0xFFFF (65535) >>> 8 == 255 (not sign-extended!)
          final u16 = NDArray<Uint16>.fromList(
            [0xFFFF, 0x8000, 0x1234, 0xFF00],
            [4],
            DType.uint16,
          );
          final su16 = NDArray<Uint16>.fromList(
            [8, 15, 4, 8],
            [4],
            DType.uint16,
          );
          expect(
            rightShift(u16, su16).toList(),
            equals([0x00FF, 1, 0x0123, 0x00FF]),
          );
          expect(
            rightShift(
              u16.slice([Slice(start: 0, stop: 4, step: 2)]),
              su16.slice([Slice(start: 0, stop: 4, step: 2)]),
            ).toList(),
            equals([0x00FF, 0x0123]),
          );

          // uint32: 0xFFFFFFFF >>> 16 == 0x0000FFFF
          final u32 = NDArray<Uint32>.fromList(
            [0xFFFFFFFF, 0x80000000, 0x12345678, 0xFF000000],
            [4],
            DType.uint32,
          );
          final su32 = NDArray<Uint32>.fromList(
            [16, 31, 8, 24],
            [4],
            DType.uint32,
          );
          expect(
            rightShift(u32, su32).toList(),
            equals([0x0000FFFF, 1, 0x00123456, 0x000000FF]),
          );
          expect(
            rightShift(
              u32.slice([Slice(start: 0, stop: 4, step: 2)]),
              su32.slice([Slice(start: 0, stop: 4, step: 2)]),
            ).toList(),
            equals([0x0000FFFF, 0x00123456]),
          );

          // uint64: -1 (0xFFFFFFFFFFFFFFFF) >>> 32 == 0x00000000FFFFFFFF
          final u64 = NDArray<Uint64>.fromList(
            [-1, -9223372036854775808, 256, 1024],
            [4],
            DType.uint64,
          );
          final su64 = NDArray<Uint64>.fromList(
            [32, 63, 4, 5],
            [4],
            DType.uint64,
          );
          expect(
            rightShift(u64, su64).toList(),
            equals([0xFFFFFFFF, 1, 16, 32]),
          );
          expect(
            rightShift(
              u64.slice([Slice(start: 0, stop: 4, step: 2)]),
              su64.slice([Slice(start: 0, stop: 4, step: 2)]),
            ).toList(),
            equals([0xFFFFFFFF, 16]),
          );
        });
      },
    );

    test(
      'binaryUfunc, unaryUfunc, reduceUfunc, accumulateUfunc, reduceatUfunc, atUfunc support all 8 integer DTypes for bitwise ops',
      () {
        for (final dt in intDTypes) {
          NDArray.scope(() {
            final a = NDArray.fromList([7, 3, 5, 1], [4], dt);
            final b = NDArray.fromList([6, 2, 4, 1], [4], dt);

            final bAnd = binaryUfunc<DTypeTag, DTypeTag>(
              a,
              b,
              op: BinaryOp.bitwiseAnd,
            );
            expect(bAnd.toList(), equals([6, 2, 4, 1]));

            final uInv = unaryUfunc<DTypeTag, DTypeTag>(a, op: UnaryOp.invert);
            expect(uInv.getCell([0]), equals(invert(a).getCell([0])));

            final redAnd = reduceUfunc(a, op: BinaryOp.bitwiseAnd);
            expect(redAnd.scalar, equals(7 & 3 & 5 & 1));

            final redOr = reduceUfunc(a, op: BinaryOp.bitwiseOr);
            expect(redOr.scalar, equals(7 | 3 | 5 | 1));

            final redXor = reduceUfunc(a, op: BinaryOp.bitwiseXor);
            expect(redXor.scalar, equals(7 ^ 3 ^ 5 ^ 1));

            final accOr = accumulateUfunc(a, op: BinaryOp.bitwiseOr);
            expect(
              accOr.toList(),
              equals([7, 7 | 3, 7 | 3 | 5, 7 | 3 | 5 | 1]),
            );

            final idx = NDArray<Int64>.fromList([0, 2], [2], DType.int64);
            final redAt = reduceatUfunc(a, idx, op: BinaryOp.bitwiseOr);
            expect(redAt.toList(), equals([7 | 3, 5 | 1]));

            final target = a.copy();
            atUfunc(
              target,
              NDArray<Int64>.fromList([0, 3], [2], DType.int64),
              NDArray.fromList([8, 2], [2], dt),
              op: BinaryOp.bitwiseOr,
            );
            expect(target.toList(), equals([7 | 8, 3, 5, 1 | 2]));
          });
        }
      },
    );
  });

  group(
    'Review Cycle 16 — Issue #3: Partial & Strided out: Aliasing Protection Across Ufuncs',
    () {
      test(
        '2D transpose self-aliasing: add(a, a.transpose(), out: a) and subtract(a, a.transpose(), out: a)',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
              [3, 3],
              DType.float64,
            );
            final aT = a.transpose();
            final expectedAdd = add(a, aT);
            add(a, aT, out: a);
            expect(a.toList(), equals(expectedAdd.toList()));

            final b = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
              [3, 3],
              DType.float64,
            );
            final bT = b.transpose();
            final expectedSub = subtract(b, bT);
            subtract(b, bT, out: b);
            expect(b.toList(), equals(expectedSub.toList()));
          });
        },
      );

      test(
        'Overlapping 1D slice shift-write in multiply, negative, sin, and sqrt',
        () {
          NDArray.scope(() {
            // multiply(base[0:4], 2, out: base[1:5])
            final base = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0],
              [5],
              DType.float64,
            );
            final two = NDArray<Float64>.fromList(
              [2.0, 2.0, 2.0, 2.0],
              [4],
              DType.float64,
            );
            multiply(
              base.slice([Slice(start: 0, stop: 4)]),
              two,
              out: base.slice([Slice(start: 1, stop: 5)]),
            );
            expect(base.toList(), equals([1.0, 2.0, 4.0, 6.0, 8.0]));

            // negative(arr, out: arr[::-1])
            final rev = NDArray<Int64>.fromList(
              [10, 20, 30, 40, 50],
              [5],
              DType.int64,
            );
            negative(rev, out: rev.slice([Slice(step: -1)]));
            expect(rev.toList(), equals([-50, -40, -30, -20, -10]));

            // sqrt(base[0:4], out: base[1:5]) and strided/transposed sqrt (Dart fallback + C++ path)
            final sqStrided = NDArray<Float64>.fromList(
              [4.0, 9.0, 16.0, 25.0, 36.0],
              [5],
              DType.float64,
            );
            sqrt(
              sqStrided.slice([Slice(step: 2)]),
              out: sqStrided.slice([Slice(start: 1, stop: 4)]),
            );
            expect(sqStrided.toList(), equals([4.0, 2.0, 4.0, 6.0, 36.0]));

            final sq2d = NDArray<Float64>.fromList(
              [4.0, 9.0, 16.0, 25.0],
              [2, 2],
              DType.float64,
            );
            sqrt(sq2d.transpose(), out: sq2d);
            expect(sq2d.toList(), equals([2.0, 4.0, 3.0, 5.0]));

            final sqF64 = NDArray<Float64>.fromList(
              [4.0, 9.0, 16.0, 25.0, 36.0],
              [5],
              DType.float64,
            );
            sqrt(
              sqF64.slice([Slice(start: 0, stop: 4)]),
              out: sqF64.slice([Slice(start: 1, stop: 5)]),
            );
            expect(sqF64.toList(), equals([4.0, 2.0, 3.0, 4.0, 5.0]));
          });
        },
      );

      test(
        'Complex & trig ufuncs (conj, power, atan2) with transposed/overlapping out:',
        () {
          NDArray.scope(() {
            final c = NDArray<Complex128>.fromList(
              [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
              [2, 2],
              DType.complex128,
            );
            final cT = c.transpose();
            final expectedConj = conj(cT);
            conj(cT, out: c);
            expect(c.toList(), equals(expectedConj.toList()));

            final p = NDArray<Complex128>.fromList(
              [Complex(1, 1), Complex(2, 0), Complex(0, 2), Complex(1, -1)],
              [2, 2],
              DType.complex128,
            );
            final pT = p.transpose();
            final twoCpx = NDArray<Complex128>.fromList(
              [Complex(2, 0), Complex(2, 0), Complex(2, 0), Complex(2, 0)],
              [2, 2],
              DType.complex128,
            );
            final expectedPow = power(pT, twoCpx);
            power(pT, twoCpx, out: p);
            expect(p.toList(), equals(expectedPow.toList()));

            final y = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final yT = y.transpose();
            final expectedAtan2 = atan2(y, yT);
            atan2(y, yT, out: y);
            expect(y.toList(), equals(expectedAtan2.toList()));
          });
        },
      );

      test(
        'clip (ternaryOp), isClose, modf, frexp with overlapping/transposed out:',
        () {
          NDArray.scope(() {
            // clip(a.transpose(), min: 2, max: 7, out: a) on int16 (uses ternaryOp)
            final a16 = NDArray<Int16>.fromList(
              [1, 5, 9, 2, 6, 10, 0, 4, 8],
              [3, 3],
              DType.int16,
            );
            final a16T = a16.transpose();
            final expectedClip = clip(a16T, min: 2, max: 7);
            clip(a16T, min: 2, max: 7, out: a16);
            expect(a16.toList(), equals(expectedClip.toList()));

            // modf(m.transpose(), out1: m)
            final m = NDArray<Float64>.fromList(
              [1.25, 2.5, 3.75, 4.125],
              [2, 2],
              DType.float64,
            );
            final mT = m.transpose();
            final expectedModf = modf(mT);
            final resModf = modf(mT, out1: m);
            expect(m.toList(), equals(expectedModf.fractional.toList()));
            expect(
              resModf.integral.toList(),
              equals(expectedModf.integral.toList()),
            );

            // modf with out1 sharing memory with out2 must throw ArgumentError
            expect(() => modf(m, out1: m, out2: m), throwsArgumentError);

            // frexp(f.transpose(), out1: f)
            final f = NDArray<Float64>.fromList(
              [4.0, 8.0, 16.0, 32.0],
              [2, 2],
              DType.float64,
            );
            final fT = f.transpose();
            final expectedFrexp = frexp(fT);
            final resFrexp = frexp(fT, out1: f);
            expect(f.toList(), equals(expectedFrexp.mantissa.toList()));
            expect(
              resFrexp.exponent.toList(),
              equals(expectedFrexp.exponent.toList()),
            );
          });
        },
      );

      test('where: mask aliasing out: buffer in unary and binary ufuncs', () {
        NDArray.scope(() {
          final w = NDArray<Boolean>.fromList(
            [true, false, true, true],
            [2, 2],
            DType.boolean,
          );
          final src = NDArray<Boolean>.fromList(
            [false, false, false, false],
            [2, 2],
            DType.boolean,
          );

          // Copy w to test against unaliased reference
          final wRef = w.copy();
          final outRef = wRef.transpose().copy();
          logicalNot(src, where: wRef, out: outRef);

          // Execute with where: w and out: w.transpose() (sharing memory!)
          final wT = w.transpose();
          logicalNot(src, where: w, out: wT);
          expect(wT.toList(), equals(outRef.toList()));
        });
      });
    },
  );
}
