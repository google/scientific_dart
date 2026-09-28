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
import 'package:ndarray/ndarray.dart';

void main() {
  final all15DTypes = [
    DType.float64,
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.complex128,
    DType.complex64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint64,
    DType.uint32,
    DType.uint16,
    DType.uint8,
    DType.boolean,
  ];

  final implementedBitwiseIntegerDTypes = [
    DType.int64,
    DType.int32,
    DType.int16,
    DType.uint8,
  ];

  final otherIntegerDTypes = [
    DType.int8,
    DType.uint64,
    DType.uint32,
    DType.uint16,
  ];

  final floatDTypes = [
    DType.float64,
    DType.float32,
    DType.float16,
    DType.bfloat16,
  ];

  NDArray<Object> makeSampleArray(
    DType dt,
    List<int> shape, {
    int seed = 1,
    bool nonZero = true,
  }) {
    final size = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (dt == DType.boolean) {
      final raw = List<bool>.generate(size, (i) => (i + seed) % 2 == 0);
      return NDArray<bool>.fromList(raw, shape, DType.boolean);
    }
    if (dt == DType.complex128) {
      final raw = List<Complex>.generate(size, (i) {
        final baseVal = nonZero ? ((i + seed) % 5) + 2 : ((i + seed) % 5);
        return Complex(baseVal.toDouble(), 1.0);
      });
      return NDArray<Complex>.fromList(raw, shape, DType.complex128);
    }
    if (dt == DType.complex64) {
      final raw = List<Complex>.generate(size, (i) {
        final baseVal = nonZero ? ((i + seed) % 5) + 2 : ((i + seed) % 5);
        return Complex(baseVal.toDouble(), 1.0);
      });
      return NDArray<Complex>.fromList(raw, shape, DType.complex64);
    }
    if (floatDTypes.contains(dt)) {
      final raw = List<double>.generate(size, (i) {
        final baseVal = nonZero ? ((i + seed) % 5) + 2 : ((i + seed) % 5);
        return baseVal.toDouble();
      });
      return NDArray<double>.fromList(raw, shape, dt as DType<double>);
    }
    final raw = List<int>.generate(size, (i) {
      final baseVal = nonZero ? ((i + seed) % 5) + 2 : ((i + seed) % 5);
      return baseVal;
    });
    return NDArray<int>.fromList(raw, shape, dt as DType<int>);
  }

  void testUnaryHelper<T extends Object>(
    NDArray<T> a,
    NDArray<T> aTrans,
    NDArray<bool> mask, {
    bool testNegative = true,
    bool testPositive = true,
    bool testRounding = false,
    bool testExpm1Log1p = false,
  }) {
    if (testNegative) {
      final neg1 = negative(a);
      expect(neg1.shape, a.shape);
      final neg2 = negative(aTrans, where: mask);
      expect(neg2.shape, a.shape);
      final outNeg = NDArray<T>.create(a.shape, a.dtype);
      final neg3 = negative(a, out: outNeg);
      expect(identical(neg3, outNeg), isTrue);
    }

    if (testPositive) {
      final pos1 = positive(a);
      expect(pos1.shape, a.shape);
      final pos2 = positive(aTrans, where: mask);
      expect(pos2.shape, a.shape);
      final outPos = NDArray<T>.create(a.shape, a.dtype);
      final pos3 = positive(a, out: outPos);
      expect(identical(pos3, outPos), isTrue);
    }

    // abs
    final abs1 = abs(a);
    expect(abs1.shape, a.shape);
    final abs2 = abs(aTrans, where: mask);
    expect(abs2.shape, a.shape);

    // square
    final sq1 = square(a);
    expect(sq1.shape, a.shape);
    final sq2 = square(aTrans, where: mask);
    expect(sq2.shape, a.shape);
    final outSq = NDArray<T>.create(a.shape, a.dtype);
    final sq3 = square(a, out: outSq);
    expect(identical(sq3, outSq), isTrue);

    // sqrt
    final sqrt1 = sqrt(a);
    expect(sqrt1.shape, a.shape);
    final sqrt2 = sqrt(aTrans, where: mask);
    expect(sqrt2.shape, a.shape);

    // reciprocal
    final rec1 = reciprocal(a);
    expect(rec1.shape, a.shape);
    final rec2 = reciprocal(aTrans, where: mask);
    expect(rec2.shape, a.shape);
    final outRec = NDArray<T>.create(a.shape, a.dtype);
    final rec3 = reciprocal(a, out: outRec);
    expect(identical(rec3, outRec), isTrue);

    // sign
    final sgn1 = sign(a);
    expect(sgn1.shape, a.shape);
    final sgn2 = sign(aTrans, where: mask);
    expect(sgn2.shape, a.shape);
    final outSgn = NDArray<T>.create(a.shape, a.dtype);
    final sgn3 = sign(a, out: outSgn);
    expect(identical(sgn3, outSgn), isTrue);

    // conj & conjugate
    final conj1 = conj(a);
    expect(conj1.shape, a.shape);
    final conj2 = conjugate(aTrans, where: mask);
    expect(conj2.shape, a.shape);
    final outConj = NDArray<T>.create(a.shape, a.dtype);
    final conj3 = conj(a, out: outConj);
    expect(identical(conj3, outConj), isTrue);

    if (testRounding) {
      final r1 = rint(a);
      expect(r1.shape, a.shape);
      final r2 = rint(aTrans, where: mask);
      expect(r2.shape, a.shape);

      final t1 = trunc(a);
      expect(t1.shape, a.shape);
      final t2 = trunc(aTrans, where: mask);
      expect(t2.shape, a.shape);

      final f1 = fix(a);
      expect(f1.shape, a.shape);
      final f2 = fix(aTrans, where: mask);
      expect(f2.shape, a.shape);

      final c1 = ceil(a);
      expect(c1.shape, a.shape);
      final c2 = ceil(aTrans, where: mask);
      expect(c2.shape, a.shape);
      final outCeil = NDArray<T>.create(a.shape, a.dtype);
      final c3 = ceil(a, out: outCeil);
      expect(identical(c3, outCeil), isTrue);

      final fl1 = floor(a);
      expect(fl1.shape, a.shape);
      final fl2 = floor(aTrans, where: mask);
      expect(fl2.shape, a.shape);
      final outFloor = NDArray<T>.create(a.shape, a.dtype);
      final fl3 = floor(a, out: outFloor);
      expect(identical(fl3, outFloor), isTrue);

      final rd1 = round(a);
      expect(rd1.shape, a.shape);
      final rd2 = round(aTrans, where: mask);
      expect(rd2.shape, a.shape);
      final outRound = NDArray<T>.create(a.shape, a.dtype);
      final rd3 = round(a, out: outRound);
      expect(identical(rd3, outRound), isTrue);
    }

    if (testExpm1Log1p) {
      final e1 = expm1(a);
      expect(e1.shape, a.shape);
      final e2 = expm1(aTrans, where: mask);
      expect(e2.shape, a.shape);

      final l1 = log1p(a);
      expect(l1.shape, a.shape);
      final l2 = log1p(aTrans, where: mask);
      expect(l2.shape, a.shape);
    }
  }

  group('Unary Operations Deep Coverage across all 15 DTypes', () {
    test('negative, positive, abs, square, sqrt, reciprocal across DTypes', () {
      NDArray.scope(() {
        final mask = NDArray<bool>.fromList(
          [true, false, true, false, true, false],
          [2, 3],
          DType.boolean,
        );

        for (final dt in all15DTypes) {
          final aContig = makeSampleArray(dt, [2, 3], seed: 1);
          final aBase = makeSampleArray(dt, [3, 2], seed: 1);
          final aTrans = aBase.transpose();

          if (dt == DType.boolean) {
            expect(() => positive(aContig), throwsUnsupportedError);
            expect(() => negative(aContig), throwsUnsupportedError);
            final conj1 = conj(aContig as NDArray<bool>);
            expect(conj1.shape, [2, 3]);
            final conj2 = conjugate(aTrans as NDArray<bool>, where: mask);
            expect(conj2.shape, [2, 3]);
          } else if (dt.isComplex) {
            testUnaryHelper<Complex>(
              aContig as NDArray<Complex>,
              aTrans as NDArray<Complex>,
              mask,
              testNegative: true,
              testPositive: true,
              testRounding: false,
              testExpm1Log1p: true,
            );
            expect(() => rint(aContig), throwsUnsupportedError);
            expect(() => trunc(aContig), throwsUnsupportedError);
            expect(() => fix(aContig), throwsUnsupportedError);
          } else if (floatDTypes.contains(dt)) {
            testUnaryHelper<double>(
              aContig as NDArray<double>,
              aTrans as NDArray<double>,
              mask,
              testNegative: true,
              testPositive: true,
              testRounding: true,
              testExpm1Log1p: true,
            );
          } else {
            testUnaryHelper<int>(
              aContig as NDArray<int>,
              aTrans as NDArray<int>,
              mask,
              testNegative: true,
              testPositive: true,
              testRounding: true,
              testExpm1Log1p: true,
            );
          }
        }
      });
    });

    test('Unary operations error handling & invalid buffer checks', () {
      NDArray.scope(() {
        final a = NDArray<double>.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final badOutShape = NDArray<double>.zeros([4], DType.float64);
        final badOutDType = NDArray<double>.zeros([3], DType.float32);

        // Incompatible shape/dtype out errors
        expect(() => sqrt(a, out: badOutShape), throwsArgumentError);
        expect(() => sqrt(a, out: badOutDType), throwsArgumentError);
        expect(() => expm1(a, out: badOutShape), throwsArgumentError);
        expect(() => expm1(a, out: badOutDType), throwsArgumentError);
        expect(() => log1p(a, out: badOutShape), throwsArgumentError);
        expect(() => log1p(a, out: badOutDType), throwsArgumentError);
        expect(() => rint(a, out: badOutShape), throwsArgumentError);
        expect(() => rint(a, out: badOutDType), throwsArgumentError);
        expect(() => trunc(a, out: badOutShape), throwsArgumentError);
        expect(() => trunc(a, out: badOutDType), throwsArgumentError);
        expect(() => square(a, out: badOutShape), throwsArgumentError);
        expect(() => square(a, out: badOutDType), throwsArgumentError);
        expect(() => reciprocal(a, out: badOutShape), throwsArgumentError);
        expect(() => reciprocal(a, out: badOutDType), throwsArgumentError);
        expect(() => positive(a, out: badOutShape), throwsArgumentError);
        expect(() => positive(a, out: badOutDType), throwsArgumentError);
        expect(() => negative(a, out: badOutShape), throwsArgumentError);
        expect(() => negative(a, out: badOutDType), throwsArgumentError);
        expect(() => abs(a, out: badOutShape), throwsArgumentError);
        expect(() => abs(a, out: badOutDType), throwsArgumentError);
        expect(() => sign(a, out: badOutShape), throwsArgumentError);
        expect(() => sign(a, out: badOutDType), throwsArgumentError);
        expect(() => conj(a, out: badOutShape), throwsArgumentError);
        expect(() => conj(a, out: badOutDType), throwsArgumentError);

        // Disposed array errors
        final dispArr = NDArray<double>.fromList(
          [1.0, 2.0],
          [2],
          DType.float64,
        );
        dispArr.dispose();
        expect(() => sqrt(dispArr), throwsStateError);
        expect(() => expm1(dispArr), throwsStateError);
        expect(() => log1p(dispArr), throwsStateError);
        expect(() => rint(dispArr), throwsStateError);
        expect(() => trunc(dispArr), throwsStateError);
        expect(() => fix(dispArr), throwsStateError);
        expect(() => square(dispArr), throwsStateError);
        expect(() => reciprocal(dispArr), throwsStateError);
        expect(() => positive(dispArr), throwsStateError);
        expect(() => negative(dispArr), throwsStateError);
        expect(() => abs(dispArr), throwsStateError);
        expect(() => sign(dispArr), throwsStateError);
        expect(() => conj(dispArr), throwsStateError);
      });
    });
  });

  group('Binary Operations Deep Coverage across all 15 DTypes', () {
    test('divmod, remainder, mod, fmod, floor_divide across Integer and Float types', () {
      NDArray.scope(() {
        final mask = NDArray<bool>.fromList(
          [true, false, true, false, true, false],
          [2, 3],
          DType.boolean,
        );

        final nonComplexTypes = [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ];

        for (final dtA in nonComplexTypes) {
          for (final dtB in nonComplexTypes) {
            final aContig = makeSampleArray(
              dtA,
              [2, 3],
              seed: 4,
              nonZero: true,
            );
            final bContig = makeSampleArray(
              dtB,
              [2, 3],
              seed: 2,
              nonZero: true,
            );

            final aBase = makeSampleArray(dtA, [3, 2], seed: 4, nonZero: true);
            final bBase = makeSampleArray(dtB, [3, 2], seed: 2, nonZero: true);
            final aTrans = aBase.transpose();
            final bTrans = bBase.transpose();

            // divmod
            final (divRes1, modRes1) = divmod(aContig, bContig);
            expect(divRes1.shape, [2, 3]);
            expect(modRes1.shape, [2, 3]);

            final (divRes2, modRes2) = divmod(aTrans, bTrans);
            expect(divRes2.shape, [2, 3]);
            expect(modRes2.shape, [2, 3]);

            // remainder & mod
            final rem1 = remainder(aContig, bContig, where: mask);
            expect(rem1.shape, [2, 3]);
            final rem2 = remainder(aTrans, bTrans);
            expect(rem2.shape, [2, 3]);
            final mod1 = mod(aContig, bContig);
            expect(mod1.shape, [2, 3]);

            // fmod
            final fmod1 = fmod(aContig, bContig, where: mask);
            expect(fmod1.shape, [2, 3]);
            final fmod2 = fmod(aTrans, bTrans);
            expect(fmod2.shape, [2, 3]);

            // floor_divide
            final fd1 = floor_divide(aContig, bContig, where: mask);
            expect(fd1.shape, [2, 3]);
            final fd2 = floor_divide(aTrans, bTrans);
            expect(fd2.shape, [2, 3]);
          }
        }
      });
    });

    test('gcd, lcm, heaviside across applicable DTypes and Strided Views', () {
      NDArray.scope(() {
        final mask = NDArray<bool>.fromList(
          [true, false, true, false, true, false],
          [2, 3],
          DType.boolean,
        );

        // gcd & lcm on integer DTypes
        for (final dtA in implementedBitwiseIntegerDTypes) {
          for (final dtB in implementedBitwiseIntegerDTypes) {
            final aContig = makeSampleArray(
              dtA,
              [2, 3],
              seed: 6,
              nonZero: true,
            );
            final bContig = makeSampleArray(
              dtB,
              [2, 3],
              seed: 4,
              nonZero: true,
            );
            final aTrans = makeSampleArray(
              dtA,
              [3, 2],
              seed: 6,
              nonZero: true,
            ).transpose();
            final bTrans = makeSampleArray(
              dtB,
              [3, 2],
              seed: 4,
              nonZero: true,
            ).transpose();

            final gcd1 = gcd(aContig, bContig, where: mask);
            expect(gcd1.shape, [2, 3]);
            final gcd2 = gcd(aTrans, bTrans);
            expect(gcd2.shape, [2, 3]);

            final lcm1 = lcm(aContig, bContig, where: mask);
            expect(lcm1.shape, [2, 3]);
            final lcm2 = lcm(aTrans, bTrans);
            expect(lcm2.shape, [2, 3]);

            // Broadcasting gcd & lcm: [2, 3] with [1, 3]
            final bBcast = makeSampleArray(dtB, [1, 3], seed: 2, nonZero: true);
            final gcdBcast = gcd(aContig, bBcast);
            expect(gcdBcast.shape, [2, 3]);
            final lcmBcast = lcm(aContig, bBcast);
            expect(lcmBcast.shape, [2, 3]);
          }
        }

        // heaviside on float and integer types
        final heavisideTypes = [
          DType.float64,
          DType.float32,
          DType.int64,
          DType.int32,
        ];
        for (final dtA in heavisideTypes) {
          for (final dtB in heavisideTypes) {
            final aContig = makeSampleArray(dtA, [2, 3], seed: 1);
            final bContig = makeSampleArray(dtB, [2, 3], seed: 2);
            final aTrans = makeSampleArray(dtA, [3, 2], seed: 1).transpose();
            final bTrans = makeSampleArray(dtB, [3, 2], seed: 2).transpose();

            final h1 = heaviside(aContig, bContig, where: mask);
            expect(h1.shape, [2, 3]);
            final h2 = heaviside(aTrans, bTrans);
            expect(h2.shape, [2, 3]);

            // Broadcasting heaviside [2, 3] with [1, 3]
            final bBcast = makeSampleArray(dtB, [1, 3], seed: 3);
            final h3 = heaviside(aContig, bBcast);
            expect(h3.shape, [2, 3]);
          }
        }
      });
    });

    test(
      'logaddexp, logaddexp2, copysign with Broadcasting & Strided Views',
      () {
        NDArray.scope(() {
          final mask = NDArray<bool>.fromList(
            [true, false, true, false, true, false],
            [2, 3],
            DType.boolean,
          );

          for (final dtA in floatDTypes) {
            for (final dtB in floatDTypes) {
              final aContig = makeSampleArray(dtA, [2, 3], seed: 1);
              final bContig = makeSampleArray(dtB, [2, 3], seed: 2);
              final aTrans = makeSampleArray(dtA, [3, 2], seed: 1).transpose();
              final bTrans = makeSampleArray(dtB, [3, 2], seed: 2).transpose();
              final bBcast = makeSampleArray(dtB, [1, 3], seed: 3);

              // logaddexp
              final lae1 = logaddexp(aContig, bContig, where: mask);
              expect(lae1.shape, [2, 3]);
              final lae2 = logaddexp(aTrans, bTrans);
              expect(lae2.shape, [2, 3]);
              final lae3 = logaddexp(aContig, bBcast);
              expect(lae3.shape, [2, 3]);

              // logaddexp2
              final lae2_1 = logaddexp2(aContig, bContig, where: mask);
              expect(lae2_1.shape, [2, 3]);
              final lae2_2 = logaddexp2(aTrans, bTrans);
              expect(lae2_2.shape, [2, 3]);
              final lae2_3 = logaddexp2(aContig, bBcast);
              expect(lae2_3.shape, [2, 3]);

              // copysign
              final cs1 = copysign(aContig, bContig, where: mask);
              expect(cs1.shape, [2, 3]);
              final cs2 = copysign(aTrans, bTrans);
              expect(cs2.shape, [2, 3]);
              final cs3 = copysign(aContig, bBcast);
              expect(cs3.shape, [2, 3]);
            }
          }
        });
      },
    );

    test('Binary operations error cases & unsupported operands', () {
      NDArray.scope(() {
        final aFloat = NDArray<double>.fromList([1.0, 2.0], [2], DType.float64);
        final bFloat = NDArray<double>.fromList([3.0, 4.0], [2], DType.float64);
        final cCplx = NDArray<Complex>.fromList(
          [Complex(1.0, 2.0)],
          [1],
          DType.complex128,
        );
        final bZeroInt = NDArray<int>.fromList([0, 2], [2], DType.int32);
        final aInt = NDArray<int>.fromList([4, 6], [2], DType.int32);

        // Complex unsupported for logaddexp, logaddexp2, heaviside, copysign, fmod
        expect(() => logaddexp(aFloat, cCplx), throwsUnsupportedError);
        expect(() => logaddexp2(cCplx, aFloat), throwsUnsupportedError);
        expect(() => heaviside(aFloat, cCplx), throwsUnsupportedError);
        expect(() => copysign(cCplx, aFloat), throwsUnsupportedError);
        expect(() => fmod(cCplx, cCplx), throwsUnsupportedError);

        // Integer division by zero
        expect(() => floor_divide(aInt, bZeroInt), throwsUnsupportedError);
        expect(() => remainder(aInt, bZeroInt), throwsUnsupportedError);
        expect(() => fmod(aInt, bZeroInt), throwsUnsupportedError);
        expect(() => divmod(aInt, bZeroInt), throwsUnsupportedError);

        // Incompatible broadcast shapes
        final badShape = NDArray<double>.zeros([5], DType.float64);
        expect(() => add(aFloat, badShape), throwsArgumentError);
        expect(() => subtract(aFloat, badShape), throwsArgumentError);
        expect(() => multiply(aFloat, badShape), throwsArgumentError);
        expect(() => divide(aFloat, badShape), throwsArgumentError);

        // Disposed array errors
        final disp = NDArray<double>.zeros([2], DType.float64);
        disp.dispose();
        expect(() => add(disp, bFloat), throwsStateError);
        expect(() => subtract(aFloat, disp), throwsStateError);
        expect(() => multiply(disp, bFloat), throwsStateError);
        expect(() => divide(aFloat, disp), throwsStateError);
        expect(() => floor_divide(disp, disp), throwsStateError);
        expect(() => remainder(disp, disp), throwsStateError);
        expect(() => fmod(disp, disp), throwsStateError);
        expect(() => heaviside(disp, disp), throwsStateError);
        expect(() => copysign(disp, disp), throwsStateError);
        expect(() => logaddexp(disp, disp), throwsStateError);
        expect(() => logaddexp2(disp, disp), throwsStateError);
      });
    });
  });

  group('Logical & Comparison Operations Deep Coverage', () {
    test(
      'logical_not across all 15 DTypes in Contiguous and Strided views',
      () {
        NDArray.scope(() {
          final mask = NDArray<bool>.fromList(
            [true, false, true, false, true, false],
            [2, 3],
            DType.boolean,
          );

          for (final dt in all15DTypes) {
            // Contiguous view [2, 3] -> hits v_to_bool_* and v_logical_not
            final aContig = makeSampleArray(dt, [2, 3], seed: 0);
            final not1 = logical_not(aContig);
            expect(not1.shape, [2, 3]);
            expect(not1.dtype, DType.boolean);

            // Non-contiguous transposed view [2, 3] -> hits s_to_bool_* and s_logical_not
            final aBase = makeSampleArray(dt, [3, 2], seed: 0);
            final aTrans = aBase.transpose();
            final not2 = logical_not(aTrans, where: mask);
            expect(not2.shape, [2, 3]);
            expect(not2.dtype, DType.boolean);

            // Out buffer reuse
            final outBuf = NDArray<bool>.zeros([2, 3], DType.boolean);
            final not3 = logical_not(aContig, out: outBuf);
            expect(identical(not3, outBuf), isTrue);
          }
        });
      },
    );

    test('logical_and, logical_or, logical_xor across all 15 DTypes with Broadcasting', () {
      NDArray.scope(() {
        final mask = NDArray<bool>.fromList(
          [true, false, true, false, true, false],
          [2, 3],
          DType.boolean,
        );

        for (final dtA in all15DTypes) {
          for (final dtB in [
            DType.boolean,
            DType.int32,
            DType.float64,
            DType.complex128,
          ]) {
            final aContig = makeSampleArray(dtA, [2, 3], seed: 1);
            final bContig = makeSampleArray(dtB, [2, 3], seed: 2);
            final aTrans = makeSampleArray(dtA, [3, 2], seed: 1).transpose();
            final bTrans = makeSampleArray(dtB, [3, 2], seed: 2).transpose();
            final bBcast = makeSampleArray(dtB, [1, 3], seed: 3);

            // Contiguous
            final land1 = logical_and(aContig, bContig);
            expect(land1.shape, [2, 3]);
            final lor1 = logical_or(aContig, bContig);
            expect(lor1.shape, [2, 3]);
            final lxor1 = logical_xor(aContig, bContig);
            expect(lxor1.shape, [2, 3]);

            // Strided transposed with mask
            final land2 = logical_and(aTrans, bTrans, where: mask);
            expect(land2.shape, [2, 3]);
            final lor2 = logical_or(aTrans, bTrans, where: mask);
            expect(lor2.shape, [2, 3]);
            final lxor2 = logical_xor(aTrans, bTrans, where: mask);
            expect(lxor2.shape, [2, 3]);

            // Broadcasting [2, 3] and [1, 3]
            final land3 = logical_and(aContig, bBcast);
            expect(land3.shape, [2, 3]);
            final lor3 = logical_or(aContig, bBcast);
            expect(lor3.shape, [2, 3]);
            final lxor3 = logical_xor(aContig, bBcast);
            expect(lxor3.shape, [2, 3]);
          }
        }
      });
    });

    test('equal, not_equal, greater, greater_equal, less, less_equal across DTypes', () {
      NDArray.scope(() {
        final mask = NDArray<bool>.fromList(
          [true, false, true, false, true, false],
          [2, 3],
          DType.boolean,
        );

        for (final dtA in all15DTypes) {
          for (final dtB in [
            DType.int32,
            DType.float64,
            DType.uint8,
            DType.boolean,
            DType.complex128,
          ]) {
            final aContig = makeSampleArray(dtA, [2, 3], seed: 1);
            final bContig = makeSampleArray(dtB, [2, 3], seed: 2);
            final aTrans = makeSampleArray(dtA, [3, 2], seed: 1).transpose();
            final bTrans = makeSampleArray(dtB, [3, 2], seed: 2).transpose();
            final bBcast = makeSampleArray(dtB, [1, 3], seed: 3);

            // Equality (supported across all types including complex)
            final eq1 = equal(aContig, bContig);
            expect(eq1.shape, [2, 3]);
            final eq2 = equal(aTrans, bTrans, where: mask);
            expect(eq2.shape, [2, 3]);
            final eq3 = equal(aContig, bBcast);
            expect(eq3.shape, [2, 3]);

            final neq1 = notEqual(aContig, bContig);
            expect(neq1.shape, [2, 3]);
            final neq2 = notEqual(aTrans, bTrans, where: mask);
            expect(neq2.shape, [2, 3]);

            // Inequalities: supported only for non-complex
            if (!dtA.isComplex && !dtB.isComplex) {
              final gt1 = greater(aContig, bContig);
              expect(gt1.shape, [2, 3]);
              final gt2 = greater(aTrans, bTrans, where: mask);
              expect(gt2.shape, [2, 3]);
              final gt3 = greater(aContig, bBcast);
              expect(gt3.shape, [2, 3]);

              final ge1 = greaterEqual(aContig, bContig);
              expect(ge1.shape, [2, 3]);
              final ge2 = greaterEqual(aTrans, bTrans, where: mask);
              expect(ge2.shape, [2, 3]);

              final lt1 = less(aContig, bContig);
              expect(lt1.shape, [2, 3]);
              final lt2 = less(aTrans, bTrans, where: mask);
              expect(lt2.shape, [2, 3]);

              final le1 = lessEqual(aContig, bContig);
              expect(le1.shape, [2, 3]);
              final le2 = lessEqual(aTrans, bTrans, where: mask);
              expect(le2.shape, [2, 3]);
            } else if (dtA.isComplex || dtB.isComplex) {
              // Inequality on complex throws UnsupportedError
              expect(() => greater(aContig, bContig), throwsUnsupportedError);
              expect(
                () => greaterEqual(aContig, bContig),
                throwsUnsupportedError,
              );
              expect(() => less(aContig, bContig), throwsUnsupportedError);
              expect(() => lessEqual(aContig, bContig), throwsUnsupportedError);
            }
          }
        }
      });
    });

    test('Logical and Comparison error cases and invalid out buffers', () {
      NDArray.scope(() {
        final a = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
        final b = NDArray<int>.fromList([1, 4, 3], [3], DType.int32);
        final badShapeOut = NDArray<bool>.zeros([5], DType.boolean);

        expect(() => logical_not(a, out: badShapeOut), throwsArgumentError);
        expect(() => equal(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => notEqual(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => greater(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => greaterEqual(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => less(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => lessEqual(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => logical_and(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => logical_or(a, b, out: badShapeOut), throwsArgumentError);
        expect(() => logical_xor(a, b, out: badShapeOut), throwsArgumentError);

        final disp = NDArray<bool>.fromList([true], [1], DType.boolean);
        disp.dispose();
        expect(() => logical_not(disp), throwsStateError);
        expect(() => logical_and(disp, disp), throwsStateError);
        expect(() => logical_or(disp, disp), throwsStateError);
        expect(() => logical_xor(disp, disp), throwsStateError);
        expect(() => equal(disp, disp), throwsStateError);
        expect(() => notEqual(disp, disp), throwsStateError);
        expect(() => greater(disp, disp), throwsStateError);
        expect(() => greaterEqual(disp, disp), throwsStateError);
        expect(() => less(disp, disp), throwsStateError);
        expect(() => lessEqual(disp, disp), throwsStateError);
      });
    });
  });

  group('Bitwise Operations on Integer Types & Strided Views', () {
    test('bitwise_and, bitwise_or, bitwise_xor, invert, left_shift, right_shift on supported integer types', () {
      NDArray.scope(() {
        final mask = NDArray<bool>.fromList(
          [true, false, true, false, true, false],
          [2, 3],
          DType.boolean,
        );

        for (final dtA in implementedBitwiseIntegerDTypes) {
          // invert
          final aContig = makeSampleArray(dtA, [2, 3], seed: 5);
          final aBase = makeSampleArray(dtA, [3, 2], seed: 5);
          final aTrans = aBase.transpose();

          final inv1 = invert(aContig);
          expect(inv1.shape, [2, 3]);
          expect(inv1.dtype, dtA);
          final inv2 = invert(aTrans, where: mask);
          expect(inv2.shape, [2, 3]);

          final outInv = NDArray.create([2, 3], dtA);
          final inv3 = invert(aContig, out: outInv);
          expect(identical(inv3, outInv), isTrue);

          // Test operations with same dtype
          final bContig = makeSampleArray(dtA, [2, 3], seed: 3);
          final bTrans = makeSampleArray(dtA, [3, 2], seed: 3).transpose();
          final bBcast = makeSampleArray(dtA, [1, 3], seed: 1);

          // bitwise_and
          final band1 = bitwise_and(aContig, bContig);
          expect(band1.shape, [2, 3]);
          final band2 = bitwise_and(aTrans, bTrans, where: mask);
          expect(band2.shape, [2, 3]);
          final band3 = bitwise_and(aContig, bBcast);
          expect(band3.shape, [2, 3]);

          // bitwise_or
          final bor1 = bitwise_or(aContig, bContig);
          expect(bor1.shape, [2, 3]);
          final bor2 = bitwise_or(aTrans, bTrans, where: mask);
          expect(bor2.shape, [2, 3]);
          final bor3 = bitwise_or(aContig, bBcast);
          expect(bor3.shape, [2, 3]);

          // bitwise_xor
          final bxor1 = bitwise_xor(aContig, bContig);
          expect(bxor1.shape, [2, 3]);
          final bxor2 = bitwise_xor(aTrans, bTrans, where: mask);
          expect(bxor2.shape, [2, 3]);
          final bxor3 = bitwise_xor(aContig, bBcast);
          expect(bxor3.shape, [2, 3]);

          // left_shift & right_shift
          final lshift1 = left_shift(aContig, bContig);
          expect(lshift1.shape, [2, 3]);
          final lshift2 = left_shift(aTrans, bTrans, where: mask);
          expect(lshift2.shape, [2, 3]);
          final lshift3 = left_shift(aContig, bBcast);
          expect(lshift3.shape, [2, 3]);

          final rshift1 = right_shift(aContig, bContig);
          expect(rshift1.shape, [2, 3]);
          final rshift2 = right_shift(aTrans, bTrans, where: mask);
          expect(rshift2.shape, [2, 3]);
          final rshift3 = right_shift(aContig, bBcast);
          expect(rshift3.shape, [2, 3]);
        }

        // Test unsupported integer types throw UnsupportedError
        for (final dt in otherIntegerDTypes) {
          final arr = makeSampleArray(dt, [2, 3]);
          expect(() => invert(arr), throwsUnsupportedError);
          expect(() => bitwise_and(arr, arr), throwsUnsupportedError);
          expect(() => bitwise_or(arr, arr), throwsUnsupportedError);
          expect(() => bitwise_xor(arr, arr), throwsUnsupportedError);
          expect(() => left_shift(arr, arr), throwsUnsupportedError);
          expect(() => right_shift(arr, arr), throwsUnsupportedError);
        }
      });
    });

    test('Bitwise operations error cases & type validation', () {
      NDArray.scope(() {
        final floatArr = NDArray<double>.fromList(
          [1.0, 2.0],
          [2],
          DType.float64,
        );
        final intArr = NDArray<int>.fromList([1, 2], [2], DType.int32);
        final boolArr = NDArray<bool>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );

        // Non-integer inputs throw ArgumentError
        expect(() => invert(floatArr), throwsArgumentError);
        expect(() => invert(boolArr), throwsArgumentError);
        expect(() => bitwise_and(floatArr, intArr), throwsArgumentError);
        expect(() => bitwise_or(intArr, floatArr), throwsArgumentError);
        expect(() => bitwise_xor(boolArr, intArr), throwsArgumentError);
        expect(() => left_shift(floatArr, intArr), throwsArgumentError);
        expect(() => right_shift(intArr, floatArr), throwsArgumentError);

        // Disposed array throws StateError
        final disp = NDArray<int>.fromList([1, 2], [2], DType.int32);
        disp.dispose();
        expect(() => invert(disp), throwsStateError);
        expect(() => bitwise_and(disp, intArr), throwsStateError);
        expect(() => bitwise_or(intArr, disp), throwsStateError);
        expect(() => bitwise_xor(disp, disp), throwsStateError);
        expect(() => left_shift(disp, disp), throwsStateError);
        expect(() => right_shift(disp, disp), throwsStateError);
      });
    });
  });

  group('Floating Point Classification & Tolerance Suite', () {
    test(
      'isnan, isinf, isfinite across Float, Complex, Int, and Bool types',
      () {
        NDArray.scope(() {
          final mask = NDArray<bool>.fromList(
            [true, false, true, false],
            [2, 2],
            DType.boolean,
          );

          // Float64 special values
          final f64Arr = NDArray<double>.fromList(
            [0.0, double.nan, double.infinity, double.negativeInfinity],
            [2, 2],
            DType.float64,
          );
          final f64Trans = f64Arr.transpose();

          final nanRes1 = isnan(f64Arr);
          expect(nanRes1.data, [false, true, false, false]);
          final nanRes2 = isnan(f64Trans, where: mask);
          expect(nanRes2.shape, [2, 2]);

          final infRes1 = isinf(f64Arr);
          expect(infRes1.data, [false, false, true, true]);
          final infRes2 = isinf(f64Trans, where: mask);
          expect(infRes2.shape, [2, 2]);

          final finRes1 = isfinite(f64Arr);
          expect(finRes1.data, [true, false, false, false]);
          final finRes2 = isfinite(f64Trans, where: mask);
          expect(finRes2.shape, [2, 2]);

          // Float32 special values
          final f32Arr = NDArray<double>.fromList(
            [1.0, double.nan, double.infinity, -2.5],
            [2, 2],
            DType.float32,
          );
          expect(isnan(f32Arr).data, [false, true, false, false]);
          expect(isinf(f32Arr).data, [false, false, true, false]);
          expect(isfinite(f32Arr).data, [true, false, false, true]);

          // Complex128 special values
          final c128Arr = NDArray<Complex>.fromList(
            [
              Complex(1.0, 2.0),
              Complex(double.nan, 0.0),
              Complex(0.0, double.infinity),
              Complex(double.negativeInfinity, double.nan),
            ],
            [2, 2],
            DType.complex128,
          );
          final c128Trans = c128Arr.transpose();
          expect(isnan(c128Arr).data, [false, true, false, true]);
          expect(isnan(c128Trans, where: mask).shape, [2, 2]);
          expect(isinf(c128Arr).data, [false, false, true, true]);
          expect(isinf(c128Trans, where: mask).shape, [2, 2]);
          expect(isfinite(c128Arr).data, [true, false, false, false]);
          expect(isfinite(c128Trans, where: mask).shape, [2, 2]);

          // Complex64 special values
          final c64Arr = NDArray<Complex>.fromList(
            [
              Complex(3.0, 4.0),
              Complex(double.nan, 1.0),
              Complex(1.0, double.infinity),
              Complex(0.0, 0.0),
            ],
            [2, 2],
            DType.complex64,
          );
          expect(isnan(c64Arr).data, [false, true, false, false]);
          expect(isinf(c64Arr).data, [false, false, true, false]);
          expect(isfinite(c64Arr).data, [true, false, false, true]);

          // Handled Integer types in floating_point.dart
          for (final dt in implementedBitwiseIntegerDTypes) {
            final intContig = makeSampleArray(dt, [2, 2]);
            final intTrans = intContig.transpose();

            expect(isnan(intContig).data, [false, false, false, false]);
            expect(isnan(intTrans, where: mask).shape, [2, 2]);
            expect(isinf(intContig).data, [false, false, false, false]);
            expect(isinf(intTrans, where: mask).shape, [2, 2]);
            expect(isfinite(intContig).data, [true, true, true, true]);
            expect(isfinite(intTrans, where: mask).shape, [2, 2]);
          }

          // Float16 / BFloat16 / other integers / boolean: check shape validity
          for (final dt in [
            DType.float16,
            DType.bfloat16,
            ...otherIntegerDTypes,
            DType.boolean,
          ]) {
            final hArr = makeSampleArray(dt, [2, 2]);
            expect(isnan(hArr).shape, [2, 2]);
            expect(isinf(hArr).shape, [2, 2]);
            expect(isfinite(hArr).shape, [2, 2]);
            expect(isfinite(hArr.transpose(), where: mask).shape, [2, 2]);
          }
        });
      },
    );

    test(
      'copysign with Subnormals, Signed Zeros, Infinities, and Out Buffers',
      () {
        NDArray.scope(() {
          final mask = NDArray<bool>.fromList(
            [true, false, true, false],
            [2, 2],
            DType.boolean,
          );

          final x1 = NDArray<double>.fromList(
            [1.0, -2.0, 3.0, -4.0],
            [2, 2],
            DType.float64,
          );
          final x2 = NDArray<double>.fromList(
            [-1.0, 1.0, -0.0, 0.0],
            [2, 2],
            DType.float64,
          );

          final res1 = copysign(x1, x2);
          expect(res1.data, [-1.0, 2.0, -3.0, 4.0]);

          final res2 = copysign(x1.transpose(), x2.transpose(), where: mask);
          expect(res2.shape, [2, 2]);

          final outBuf = NDArray<double>.zeros([2, 2], DType.float64);
          final res3 = copysign(x1, x2, out: outBuf);
          expect(identical(res3, outBuf), isTrue);

          // Float32 copysign
          final f32_1 = NDArray<double>.fromList(
            [5.0, -6.0],
            [2],
            DType.float32,
          );
          final f32_2 = NDArray<double>.fromList(
            [-1.0, 1.0],
            [2],
            DType.float32,
          );
          final f32Res = copysign(f32_1, f32_2);
          expect(f32Res.data, [-5.0, 6.0]);

          // Integer copysign
          final int1 = NDArray<int>.fromList([10, -20], [2], DType.int32);
          final int2 = NDArray<int>.fromList([-1, 1], [2], DType.int32);
          final intRes = copysign(int1, int2);
          expect(intRes.data, [-10, 20]);
        });
      },
    );

    test(
      'isClose & allClose with various tolerances, equalNan, Infs, and Complex',
      () {
        NDArray.scope(() {
          final a = NDArray<double>.fromList(
            [1.0, 2.0, double.nan, double.infinity],
            [4],
            DType.float64,
          );
          final b = NDArray<double>.fromList(
            [1.000001, 2.0, double.nan, double.infinity],
            [4],
            DType.float64,
          );

          // isClose with default tolerances
          final close1 = isClose(a, b, equalNan: true);
          expect(close1.data, [true, true, true, true]);
          expect(allClose(a, b, equalNan: true), isTrue);

          final close2 = isClose(a, b, equalNan: false);
          expect(close2.data, [true, true, false, true]);
          expect(allClose(a, b, equalNan: false), isFalse);

          // Custom tight tolerance
          final close3 = isClose(a, b, rtol: 1e-9, atol: 1e-9, equalNan: true);
          expect(close3.data, [false, true, true, true]);

          // Complex isClose
          final cA = NDArray<Complex>.fromList(
            [Complex(1.0, 2.0), Complex(double.nan, 1.0)],
            [2],
            DType.complex128,
          );
          final cB = NDArray<Complex>.fromList(
            [Complex(1.000001, 2.0), Complex(double.nan, 1.0)],
            [2],
            DType.complex128,
          );
          expect(allClose(cA, cB, equalNan: true), isTrue);

          // Mixed real and complex isClose
          final rA = NDArray<double>.fromList([1.0, 2.0], [2], DType.float64);
          final cB2 = NDArray<Complex>.fromList(
            [Complex(1.0, 0.0), Complex(2.0, 0.0)],
            [2],
            DType.complex128,
          );
          expect(allClose(rA, cB2), isTrue);
          expect(allClose(cB2, rA), isTrue);

          // Strided transposed isClose with where mask and out buffer
          final a2D = NDArray<double>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b2D = NDArray<double>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final mask = NDArray<bool>.fromList(
            [true, false, true, false],
            [2, 2],
            DType.boolean,
          );
          final outClose = NDArray<bool>.zeros([2, 2], DType.boolean);
          final resClose = isClose(
            a2D.transpose(),
            b2D.transpose(),
            where: mask,
            out: outClose,
          );
          expect(identical(resClose, outClose), isTrue);
        });
      },
    );
  });
}
