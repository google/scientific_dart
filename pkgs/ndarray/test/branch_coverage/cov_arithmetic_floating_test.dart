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

import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

NDArray<T> _arr<T extends DTypeTag>(List<Object?> values, DType<T> dtype) =>
    NDArray<T>.fromList(values, [values.length], dtype);

List<dynamic> _flat(NDArray<DTypeTag> a) => a.asAnySpec.toList();

DType<DTypeTag> _expectedMixedOutputDType(
  DType<DTypeTag> a,
  DType<DTypeTag> b, {
  required bool isDivide,
}) {
  if (!isDivide && a == DType.boolean && b == DType.boolean) {
    return DType.boolean;
  }
  if (a == DType.complex128 || b == DType.complex128) {
    return DType.complex128;
  }
  if (a == DType.complex64 || b == DType.complex64) {
    return (a == DType.float64 ||
            b == DType.float64 ||
            a == DType.int64 ||
            b == DType.int64)
        ? DType.complex128
        : DType.complex64;
  }
  if (a == DType.float64 || b == DType.float64) {
    return DType.float64;
  }
  if (a == DType.float32 || b == DType.float32) {
    if (a == DType.int64 ||
        b == DType.int64 ||
        a == DType.int32 ||
        b == DType.int32) {
      return DType.float64;
    }
    return DType.float32;
  }
  if (isDivide) {
    return DType.float64;
  }
  if (a == DType.int64 || b == DType.int64) {
    return DType.int64;
  }
  if (a == DType.int32 || b == DType.int32) {
    return DType.int32;
  }
  if (a == DType.int16 || b == DType.int16) {
    return DType.int16;
  }
  return DType.uint8;
}

NDArray<DTypeTag> _makeSample(DType<DTypeTag> dt, {bool strided = false}) {
  final base = switch (dt) {
    DType.boolean => _arr([true, false, true, true], DType.boolean),
    DType.complex128 => _arr([
      Complex(1.0, 2.0),
      Complex(3.0, 4.0),
      Complex(2.0, -1.0),
      Complex(4.0, 1.0),
    ], DType.complex128),
    DType.complex64 => _arr([
      Complex(1.0, 2.0),
      Complex(3.0, 4.0),
      Complex(2.0, -1.0),
      Complex(4.0, 1.0),
    ], DType.complex64),
    _ => _arr([1, 2, 3, 4], dt),
  };
  if (!strided) return base;
  return base.slice([const Slice(start: 0, stop: 4, step: 2)]);
}

void main() {
  group('floating_point.dart branch coverage', () {
    test('isnan, isinf, isfinite error, out aliasing, and dtype paths', () {
      NDArray.scope(() {
        final a = _arr([1.0, double.nan, double.infinity], DType.float64);
        final outBadShape = NDArray.zeros([2], DType.boolean);
        final mask = _arr([true, false, true], DType.boolean);

        expect(() => isnan(a, out: outBadShape), throwsArgumentError);
        expect(() => isinf(a, out: outBadShape), throwsArgumentError);
        expect(() => isfinite(a, out: outBadShape), throwsArgumentError);

        // sharesMemory(where, out) with where != null
        final maskAlias1 = _arr([true, false, true], DType.boolean);
        isnan(a, where: maskAlias1, out: maskAlias1);
        final maskAlias2 = _arr([true, false, true], DType.boolean);
        isinf(a, where: maskAlias2, out: maskAlias2);
        final maskAlias3 = _arr([true, false, true], DType.boolean);
        isfinite(a, where: maskAlias3, out: maskAlias3);

        // sharesMemory(a, out) with where == null (boolean input)
        final boolAlias1 = _arr([true, false, true], DType.boolean);
        isnan(boolAlias1, out: boolAlias1);
        final boolAlias2 = _arr([true, false, true], DType.boolean);
        isinf(boolAlias2, out: boolAlias2);
        final boolAlias3 = _arr([true, false, true], DType.boolean);
        isfinite(boolAlias3, out: boolAlias3);

        // Strided and fallback dtypes (float16, bfloat16, complex, int)
        for (final dt in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
          DType.int32,
        ]) {
          final arr = dt.isInteger
              ? _arr([0, 1, 2, 3], dt)
              : _arr([0.0, 1.0, double.nan, double.infinity], dt);
          final strided = arr.slice([const Slice(start: 0, stop: 4, step: 2)]);
          expect(
            isnan(
              arr,
              where: _arr([true, true, true, false], DType.boolean),
            ).shape,
            [4],
          );
          expect(isnan(strided).shape, [2]);
          expect(isinf(arr).shape, [4]);
          expect(isinf(strided).shape, [2]);
          expect(isfinite(arr).shape, [4]);
          expect(isfinite(strided).shape, [2]);
        }

        final c128 = _arr([
          Complex(1.0, double.nan),
          Complex(double.infinity, 0.0),
          Complex(1.0, 2.0),
        ], DType.complex128);
        final c64 = _arr([
          Complex(1.0, double.nan),
          Complex(double.infinity, 0.0),
          Complex(1.0, 2.0),
        ], DType.complex64);
        expect(
          isnan(c128.slice([const Slice(start: 0, stop: 3, step: 2)])).shape,
          [2],
        );
        expect(
          isnan(c64.slice([const Slice(start: 0, stop: 3, step: 2)])).shape,
          [2],
        );
        expect(
          isinf(c64.slice([const Slice(start: 0, stop: 3, step: 2)])).shape,
          [2],
        );
        expect(
          isfinite(c128.slice([const Slice(start: 0, stop: 3, step: 2)])).shape,
          [2],
        );
        expect(
          isfinite(c64.slice([const Slice(start: 0, stop: 3, step: 2)])).shape,
          [2],
        );
        expect(mask.isDisposed, isFalse);
      });
    });

    test('copysign and copysignAs branches', () {
      NDArray.scope(() {
        final f64 = _arr([1.0, -2.0], DType.float64);
        final badOut = NDArray.zeros([3], DType.float64);
        expect(() => copysign(f64, f64, out: badOut), throwsArgumentError);

        final csAlias = _arr([1.0, -2.0], DType.float64);
        final csMask = _arr([true, false], DType.boolean);
        copysign(csAlias, f64, where: csMask, out: csAlias);
        copysign(csAlias, f64, out: csAlias);

        // Fallback dtype (float16) with +0.0 and -0.0 in x2
        final f16a = _arr([1.0, -2.0, 3.0, -4.0], DType.float16);
        final f16b = _arr([0.0, -0.0, -1.0, 1.0], DType.float16);
        final res16 = copysign(f16a, f16b);
        expect(_flat(res16), [1.0, -2.0, -3.0, 4.0]);

        // uint64 and boolean in copysign
        final u64 = _arr([1, 2], DType.uint64);
        expect(_flat(copysign(u64, u64)), [1, 2]);
        final bArr = _arr([true, false], DType.boolean);
        expect(_flat(copysign(bArr, bArr)), [true, false]);

        // copysignAs errors and paths
        final cpx = _arr([Complex(1, 2)], DType.complex128);
        expect(
          () => copysignAs(cpx, f64, DType.float64),
          throwsUnsupportedError,
        );
        expect(
          () => copysignAs(f64, cpx, DType.float64),
          throwsUnsupportedError,
        );
        expect(
          () => copysignAs(f64, f64, DType.complex128),
          throwsUnsupportedError,
        );
        final disposed = _arr([1.0], DType.float64)..dispose();
        expect(
          () => copysignAs(disposed, f64, DType.float64),
          throwsStateError,
        );
        expect(
          () => copysignAs(f64, f64, DType.float64, out: badOut),
          throwsArgumentError,
        );
        final out64 = NDArray.zeros([2], DType.float64);
        final f32 = _arr([-1.0, 2.0], DType.float32);
        copysignAs(f64, f32, DType.float64, out: out64);
        expect(_flat(out64), [-1.0, 2.0]);
        expect(_flat(copysignAs(f64, f64, DType.float64)), [1.0, -2.0]);
      });
    });

    test('isClose and allClose contiguous and fallback branches', () {
      NDArray.scope(() {
        final a = _arr([1.0, 2.0], DType.float64);
        final disp = _arr([true, false], DType.boolean)..dispose();
        expect(() => isClose(a, a, where: disp), throwsStateError);
        expect(
          () => isClose(a, a, out: NDArray.zeros([3], DType.boolean)),
          throwsArgumentError,
        );
        expect(() => allClose(disp, a), throwsStateError);

        final mask = _arr([true, false, true], DType.boolean);
        final dtypes = <DType<DTypeTag>>[
          DType.float64,
          DType.float32,
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ];
        for (final dt in dtypes) {
          final x = _arr([1, 2, 3], dt);
          final y = _arr([1, 2, 9], dt);
          expect(_flat(isClose(x, y)), [true, true, false]);
          expect(_flat(isClose(x, y, where: mask)), [true, false, false]);
          expect(allClose(x, x), isTrue);
          expect(allClose(x, y), isFalse);
        }

        // Mixed (float64, float32) and (float32, float64) + NaN/Inf in _isCloseDouble
        final f64 = _arr([1.0, 2.0, 3.0], DType.float64);
        final f32Same = _arr([1.0, 2.0, 3.0], DType.float32);
        final f32Diff = _arr([1.0, 2.0, 9.0], DType.float32);
        expect(_flat(isClose(f64, f32Diff)), [true, true, false]);
        expect(_flat(isClose(f64, f32Diff, where: mask)), [true, false, false]);
        expect(_flat(isClose(f32Same, f64)), [true, true, true]);
        expect(_flat(isClose(f32Diff, f64, where: mask)), [true, false, false]);
        expect(allClose(f64, f32Same), isTrue);
        expect(allClose(f64, f32Diff), isFalse);
        expect(allClose(f32Same, f64), isTrue);
        expect(allClose(f32Diff, f64), isFalse);

        final f64NanInf = _arr([
          double.nan,
          double.infinity,
          1.0,
        ], DType.float64);
        final f32NanInf = _arr([
          double.nan,
          double.infinity,
          double.negativeInfinity,
        ], DType.float32);
        expect(_flat(isClose(f64NanInf, f32NanInf, equalNan: true)), [
          true,
          true,
          false,
        ]);
        expect(_flat(isClose(f64NanInf, f32NanInf, rtol: double.infinity)), [
          false,
          true,
          false,
        ]);
        final fastAliasOut = _arr([true, false, true], DType.boolean);
        isClose(f64, f64, where: fastAliasOut, out: fastAliasOut);
        expect(_flat(fastAliasOut), [true, false, true]);

        // Fallback path with useTempOut (non-contiguous out & aliasing where)
        final f16a = _arr([1.0, 2.0], DType.float16);
        final f16b = _arr([1.0, 5.0], DType.float16);
        final whereAlias = _arr([true, true], DType.boolean);
        isClose(f16a, f16b, where: whereAlias, out: whereAlias);
        expect(_flat(whereAlias), [true, false]);
        final fullOut = NDArray.zeros([4], DType.boolean);
        final stridedOut = fullOut.slice([
          const Slice(start: 0, stop: 4, step: 2),
        ]);
        isClose(f16a, f16b, out: stridedOut);
        expect(_flat(stridedOut), [true, false]);

        // Fallback diff branches: uint64, Complex, signed int opposite signs
        final u64a = _arr([1, 10, -1], DType.uint64);
        final u64b = _arr([1, 20, 5], DType.uint64);
        expect(_flat(isClose(u64a, u64b)), [true, false, false]);
        expect(allClose(u64a, u64a), isTrue);
        expect(allClose(u64a, u64b), isFalse);

        // Strided int opposite signs in fallback
        final sInt1 = _arr([
          -5,
          5,
          10,
          10,
        ], DType.int32).slice([const Slice(start: 0, stop: 4, step: 2)]);
        final sInt2 = _arr([
          5,
          -5,
          10,
          10,
        ], DType.int32).slice([const Slice(start: 0, stop: 4, step: 2)]);
        expect(_flat(isClose(sInt1, sInt2)), [false, true]);

        final i64 = _arr([1, -5, 5], DType.int64);
        final cpx = _arr([
          Complex(1.0, 0.0),
          Complex(2.0, 1.0),
          Complex(3.0, 0.0),
        ], DType.complex128);
        expect(_flat(isClose(u64a, cpx)), [true, false, false]);
        expect(_flat(isClose(cpx, u64a)), [true, false, false]);
        expect(_flat(isClose(i64, cpx)), [true, false, false]);
        expect(_flat(isClose(cpx, i64)), [true, false, false]);
      });
    });

    test(
      'modf and frexp record extensions, errors, aliasing, and special values',
      () {
        NDArray.scope(() {
          final x = _arr([
            1.5,
            -2.25,
            0.0,
            -0.0,
            double.nan,
            double.infinity,
            double.negativeInfinity,
            5e-324,
          ], DType.float64);
          final mRes = modf(x);
          expect(mRes.$1.shape, [8]);
          expect(mRes.$2.shape, [8]);
          mRes.dispose();
          expect(mRes.fractional.isDisposed, isTrue);

          final fRes = frexp(x);
          expect(fRes.$1.shape, [8]);
          expect(fRes.$2.shape, [8]);
          fRes.dispose();
          expect(fRes.mantissa.isDisposed, isTrue);

          // Disposed & Complex & invalid out errors
          final disp = _arr([1.0], DType.float64)..dispose();
          final cpx = _arr([Complex(1, 2)], DType.complex128);
          expect(() => modf(x, where: disp), throwsStateError);
          expect(() => modf(cpx.asAnySpec), throwsUnsupportedError);
          expect(() => frexp(x, where: disp), throwsStateError);
          expect(() => frexp(cpx.asAnySpec), throwsUnsupportedError);

          final f32Out = NDArray.zeros([8], DType.float32);
          final badShapeOut = NDArray.zeros([2], DType.float64);
          final badIntOut = NDArray.zeros([2], DType.int32);
          expect(() => modf(x.asAnySpec, out1: f32Out), throwsArgumentError);
          expect(() => modf(x, out1: badShapeOut), throwsArgumentError);
          expect(() => modf(x, out2: badShapeOut), throwsArgumentError);
          expect(() => modf(x, out1: fRes.$1, out2: fRes.$1), throwsStateError);
          final o1 = NDArray.zeros([8], DType.float64);
          expect(() => modf(x, out1: o1, out2: o1), throwsArgumentError);

          expect(() => frexp(x.asAnySpec, out1: f32Out), throwsArgumentError);
          expect(() => frexp(x, out1: badShapeOut), throwsArgumentError);
          expect(() => frexp(x, out2: badIntOut), throwsArgumentError);
          final expPtrAlias = NDArray<Int32>.fromPointer(o1.pointer.cast(), [
            8,
          ], DType.int32);
          expect(
            () => frexp(x, out1: o1, out2: expPtrAlias),
            throwsArgumentError,
          );

          // useTemp1 & useTemp2 in modf and frexp (out1/out2 aliases x or where, with and without where)
          final xAlias1 = _arr([1.5, -2.5], DType.float64);
          final out1Clean = NDArray.zeros([2], DType.float64);
          final mask2 = _arr([true, false], DType.boolean);
          modf(xAlias1, out1: xAlias1, out2: out1Clean, where: mask2);
          modf(xAlias1, out1: xAlias1, out2: out1Clean);
          modf(xAlias1, out1: out1Clean, out2: xAlias1, where: mask2);
          expect(_flat(xAlias1)[0], 0.0);

          final xAlias2 = _arr([1.5, -2.5], DType.float64);
          modf(xAlias2, out1: out1Clean, out2: xAlias2);
          expect(_flat(xAlias2), [1.0, -2.0]);

          final f64AliasFrexp = _arr([1.5, -2.5], DType.float64);
          final i32OutClean = NDArray.zeros([2], DType.int32);
          frexp(
            f64AliasFrexp,
            out1: f64AliasFrexp,
            out2: i32OutClean,
            where: mask2,
          );
          frexp(f64AliasFrexp, out1: f64AliasFrexp, out2: i32OutClean);
          final i32Input = _arr([4, 8], DType.int32);
          frexp(i32Input, out1: out1Clean, out2: i32Input, where: mask2);
          final i32Input2 = _arr([4, 8], DType.int32);
          frexp(i32Input2, out1: out1Clean, out2: i32Input2);
          expect(_flat(i32Input2), [3, 4]);

          // uint64, boolean, float16 in modf and frexp
          final u64 = _arr([0, 3, -1], DType.uint64);
          final bArr = _arr([true, false], DType.boolean);
          final f16 = _arr([
            1.5,
            -2.5,
            0.0,
            -0.0,
            double.nan,
            double.infinity,
            double.negativeInfinity,
          ], DType.float16);
          expect(modf(u64).$2.shape, [3]);
          expect(_flat(modf(bArr).$1), [0.0, 0.0]);
          expect(modf(f16).$1.shape, [7]);
          expect(frexp(u64).$2.shape, [3]);
          expect(_flat(frexp(bArr).$1), [0.5, 0.0]);
          expect(frexp(f16).$1.shape, [7]);
        });
      },
    );

    test('arrayEqual branches', () {
      NDArray.scope(() {
        final disp = _arr([1], DType.int32)..dispose();
        final a = _arr([1, 2], DType.int32);
        expect(() => arrayEqual(disp, a), throwsStateError);
        expect(arrayEqual(a, _arr([1, 2, 3], DType.int64)), isFalse);
        expect(
          arrayEqual(
            NDArray.zeros([0], DType.int32),
            NDArray.zeros([0], DType.float64),
          ),
          isTrue,
        );

        final u64Pos = _arr([1, 2], DType.uint64);
        final u64NegBits = _arr([-1, 2], DType.uint64);
        final i64Pos = _arr([1, 2], DType.int64);
        final i64Neg = _arr([-1, 2], DType.int64);
        final f64Pos = _arr([1.0, 2.0], DType.float64);
        expect(arrayEqual(u64Pos, i64Pos), isTrue);
        expect(arrayEqual(u64Pos, i64Neg), isFalse);
        expect(arrayEqual(u64NegBits, i64Pos), isFalse);
        expect(arrayEqual(i64Pos, u64Pos), isTrue);
        expect(arrayEqual(i64Neg, u64Pos), isFalse);
        expect(arrayEqual(i64Pos, u64NegBits), isFalse);
        expect(arrayEqual(u64Pos, f64Pos), isTrue);
        expect(arrayEqual(f64Pos, u64Pos), isTrue);

        final c1 = _arr([Complex(1.0, double.nan)], DType.complex128);
        final c2 = _arr([Complex(1.0, double.nan)], DType.complex128);
        final c3 = _arr([Complex(1.0, 2.0)], DType.complex128);
        expect(arrayEqual(c1, c2, equalNan: false), isFalse);
        expect(arrayEqual(c1, c2, equalNan: true), isTrue);
        expect(arrayEqual(c1, c3, equalNan: true), isFalse);
        expect(arrayEqual(c3, _arr([1.0], DType.float64)), isFalse);

        final n1 = _arr([double.nan, 2.0], DType.float64);
        final n2 = _arr([double.nan, 2.0], DType.float32);
        expect(arrayEqual(n1, n2, equalNan: false), isFalse);
        expect(arrayEqual(n1, n2, equalNan: true), isTrue);
      });
    });
  });

  group(
    'arithmetic.dart mixed-dtype kernel matrix (addAs, subtractAs, multiplyAs, divideAs)',
    () {
      final kernelDTypes = <DType<DTypeTag>>[
        DType.float64,
        DType.float32,
        DType.int64,
        DType.int32,
        DType.int16,
        DType.uint8,
        DType.boolean,
        DType.complex128,
        DType.complex64,
      ];

      test('all 9x9 contiguous and strided pairs hit native mixed kernels', () {
        NDArray.scope(() {
          final contigA = {for (final dt in kernelDTypes) dt: _makeSample(dt)};
          final contigB = {for (final dt in kernelDTypes) dt: _makeSample(dt)};
          final stridedA = {
            for (final dt in kernelDTypes) dt: _makeSample(dt, strided: true),
          };
          final stridedB = {
            for (final dt in kernelDTypes) dt: _makeSample(dt, strided: true),
          };

          for (final dtA in kernelDTypes) {
            for (final dtB in kernelDTypes) {
              final target = _expectedMixedOutputDType(
                dtA,
                dtB,
                isDivide: false,
              );
              final divTarget = _expectedMixedOutputDType(
                dtA,
                dtB,
                isDivide: true,
              );

              final ca = contigA[dtA]!;
              final cb = contigB[dtB]!;
              final sa = stridedA[dtA]!;
              final sb = stridedB[dtB]!;

              expect(addAs(ca, cb, target).dtype, target);
              expect(addAs(sa, sb, target).dtype, target);

              if (target != DType.boolean) {
                expect(subtractAs(ca, cb, target).dtype, target);
                expect(subtractAs(sa, sb, target).dtype, target);
              }

              expect(multiplyAs(ca, cb, target).dtype, target);
              expect(multiplyAs(sa, sb, target).dtype, target);

              expect(divideAs(ca, cb, divTarget).dtype, divTarget);
              expect(divideAs(sa, sb, divTarget).dtype, divTarget);
            }
          }

          // Special float32 / complex64 targetDType branches for (f32, i64/i32) and (c64, i64)
          for (final intDt in [DType.int64, DType.int32]) {
            for (final isStrided in [false, true]) {
              final f32 = isStrided
                  ? stridedA[DType.float32]!
                  : contigA[DType.float32]!;
              final iArr = isStrided ? stridedA[intDt]! : contigA[intDt]!;
              expect(addAs(f32, iArr, DType.float32).dtype, DType.float32);
              expect(addAs(iArr, f32, DType.float32).dtype, DType.float32);
              expect(subtractAs(f32, iArr, DType.float32).dtype, DType.float32);
              expect(subtractAs(iArr, f32, DType.float32).dtype, DType.float32);
              expect(multiplyAs(f32, iArr, DType.float32).dtype, DType.float32);
              expect(multiplyAs(iArr, f32, DType.float32).dtype, DType.float32);
              expect(divideAs(f32, iArr, DType.float32).dtype, DType.float32);
              expect(divideAs(iArr, f32, DType.float32).dtype, DType.float32);
            }
          }
          for (final isStrided in [false, true]) {
            final c64 = isStrided
                ? stridedA[DType.complex64]!
                : contigA[DType.complex64]!;
            final i64 = isStrided
                ? stridedA[DType.int64]!
                : contigA[DType.int64]!;
            expect(addAs(c64, i64, DType.complex64).dtype, DType.complex64);
            expect(addAs(i64, c64, DType.complex64).dtype, DType.complex64);
            expect(
              subtractAs(c64, i64, DType.complex64).dtype,
              DType.complex64,
            );
            expect(
              subtractAs(i64, c64, DType.complex64).dtype,
              DType.complex64,
            );
            expect(
              multiplyAs(c64, i64, DType.complex64).dtype,
              DType.complex64,
            );
            expect(
              multiplyAs(i64, c64, DType.complex64).dtype,
              DType.complex64,
            );
            expect(divideAs(c64, i64, DType.complex64).dtype, DType.complex64);
            expect(divideAs(i64, c64, DType.complex64).dtype, DType.complex64);
          }
        });
      });

      test('fallback dtypes and divideAs integer/boolean target branches', () {
        NDArray.scope(() {
          final u16 = _arr([4, 6], DType.uint16);
          final i8 = _arr([2, 3], DType.int8);
          expect(_flat(addAs(u16, i8, DType.int32)), [6, 9]);
          expect(_flat(subtractAs(u16, i8, DType.int32)), [2, 3]);
          expect(_flat(multiplyAs(u16, i8, DType.int32)), [8, 18]);

          // divideAs into integer/boolean dtype
          final outI32 = NDArray.zeros([2], DType.int32);
          divideAs(u16, i8, DType.int32, out: outI32);
          expect(_flat(outI32), [2, 2]);
          expect(_flat(divideAs(u16, i8, DType.boolean)), [true, true]);

          final f64 = _arr([5.0, 6.0], DType.float64);
          final f32 = _arr([2.0, 3.0], DType.float32);
          expect(_flat(divideAs(f64, f32, DType.int32)), [2, 2]);
          divideAs(f64, f32, DType.int32, out: outI32);
          expect(_flat(outI32), [2, 2]);

          expect(
            () => divideAs(
              u16,
              i8,
              DType.int32,
              out: NDArray.zeros([3], DType.int32),
            ),
            throwsArgumentError,
          );
          final cpx = _arr([Complex(2, 0), Complex(6, 0)], DType.complex128);
          expect(_flat(divideAs(cpx, i8, DType.int32)), [1, 2]);
          expect(_flat(divideAs(i8, cpx, DType.int32)), [1, 0]);
        });
      });
    },
  );

  group('arithmetic.dart remaining operations branch coverage', () {
    test('expm1, log1p, logaddexp, logaddexp2, and *As variants', () {
      NDArray.scope(() {
        // expm1 and log1p with float16, uint64, boolean (small and normal values)
        for (final dt in [
          DType.float16,
          DType.uint64,
          DType.int32,
          DType.boolean,
        ]) {
          final arr = dt == DType.boolean
              ? _arr([false, true], DType.boolean)
              : _arr([0, 1], dt);
          expect(expm1(arr.asAnySpec).shape, [2]);
          expect(log1p(arr.asAnySpec).shape, [2]);
        }
        final smallF16 = _arr([1e-6, 1.0], DType.float16);
        expect(expm1(smallF16).shape, [2]);
        expect(log1p(smallF16).shape, [2]);

        // logaddexp & logaddexp2 fallback (float16, uint64, boolean) & edge cases
        final f16a = _arr([
          double.nan,
          double.negativeInfinity,
          1.0,
          0.0,
          2.0,
        ], DType.float16);
        final f16b = _arr([
          1.0,
          double.negativeInfinity,
          1.0,
          20.0,
          10.0,
        ], DType.float16);
        expect(logaddexp(f16a, f16b).shape, [5]);
        expect(logaddexp2(f16a, f16b).shape, [5]);

        final u64 = _arr([0, 1], DType.uint64);
        final bArr = _arr([false, true], DType.boolean);
        expect(logaddexp(u64, u64).shape, [2]);
        expect(logaddexp(bArr, bArr).shape, [2]);
        expect(logaddexp2(u64, u64).shape, [2]);
        expect(logaddexp2(bArr, bArr).shape, [2]);

        final badOut = NDArray.zeros([3], DType.float64);
        expect(() => logaddexp(u64, u64, out: badOut), throwsArgumentError);
        expect(() => logaddexp2(u64, u64, out: badOut), throwsArgumentError);

        // logaddexpAs and logaddexp2As
        final f64 = _arr([1.0, 2.0], DType.float64);
        final f32 = _arr([1.0, 2.0], DType.float32);
        final cpx = _arr([Complex(1, 2), Complex(3, 4)], DType.complex128);
        expect(
          () => logaddexpAs(cpx, f64, DType.float64),
          throwsUnsupportedError,
        );
        expect(
          () => logaddexp2As(cpx, f64, DType.float64),
          throwsUnsupportedError,
        );
        expect(
          () => logaddexpAs(f64, f32, DType.float64, out: badOut),
          throwsArgumentError,
        );
        expect(
          () => logaddexp2As(f64, f32, DType.float64, out: badOut),
          throwsArgumentError,
        );

        final out32 = NDArray.zeros([2], DType.float32);
        final out64 = NDArray.zeros([2], DType.float64);
        logaddexpAs(f64, f32, DType.float32, out: out32);
        logaddexpAs(f64, f32, DType.float64, out: out64);
        logaddexp2As(f64, f32, DType.float32, out: out32);
        logaddexp2As(f64, f32, DType.float64, out: out64);
        expect(out32.shape, [2]);
        expect(out64.shape, [2]);
      });
    });

    test(
      'rint, square, reciprocal, positive, power, floorDivide, remainder, fmod, divmod',
      () {
        NDArray.scope(() {
          // rint fallback (float16, uint64, boolean)
          final rintF16 = _arr([
            double.infinity,
            double.nan,
            1.2,
            1.8,
            2.5,
            3.5,
          ], DType.float16);
          expect(_flat(rint(rintF16)).sublist(2), [1.0, 2.0, 2.0, 4.0]);
          expect(_flat(rint(_arr([1, 2], DType.uint64))), [1.0, 2.0]);
          expect(_flat(rint(_arr([true, false], DType.boolean))), [1.0, 0.0]);

          // square strided boolean
          final sBool = _arr([
            true,
            false,
            true,
            false,
          ], DType.boolean).slice([const Slice(start: 0, stop: 4, step: 2)]);
          expect(_flat(square(sBool)), [true, true]);

          // reciprocal strided int division by zero & fallback int division by zero
          final sIntZero = _arr([
            1,
            2,
            0,
            4,
          ], DType.int32).slice([const Slice(start: 0, stop: 4, step: 2)]);
          expect(() => reciprocal(sIntZero), throwsUnsupportedError);
          expect(
            () => reciprocal(_arr([0], DType.int8)),
            throwsUnsupportedError,
          );
          expect(_flat(reciprocal(_arr([1, 2], DType.uint64))), [1, 0]);
          expect(_flat(reciprocal(_arr([2.0, 4.0], DType.float16))), [
            0.5,
            0.25,
          ]);

          // positive aliasing
          final posArr = _arr([1, 2], DType.int32);
          final posMask = _arr([true, false], DType.boolean);
          positive(posArr, out: posArr);
          positive(posArr, where: posMask, out: posArr);
          expect(_flat(posArr), [1, 2]);

          // power 0-D negative integer exponent & powerAs same-dtype
          final base0d = NDArray.scalar(2, dtype: DType.int32);
          final exp0dNeg = NDArray.scalar(-1, dtype: DType.int32);
          expect(() => power(base0d, exp0dNeg), throwsArgumentError);
          final disp = _arr([1], DType.int32)..dispose();
          expect(() => powerAs(disp, posArr, DType.int32), throwsStateError);
          expect(_flat(powerAs(posArr, posArr, DType.int32)), [1, 4]);

          // Strided integer division by zero in floorDivide, remainder, fmod
          for (final dt in [DType.int64, DType.int32]) {
            final sA = _arr([
              4,
              5,
              6,
              7,
            ], dt).slice([const Slice(start: 0, stop: 4, step: 2)]).asAnySpec;
            final sZero = _arr([
              2,
              1,
              0,
              1,
            ], dt).slice([const Slice(start: 0, stop: 4, step: 2)]).asAnySpec;
            expect(() => floorDivide(sA, sZero), throwsUnsupportedError);
            expect(() => remainder(sA, sZero), throwsUnsupportedError);
            expect(() => fmod(sA, sZero), throwsUnsupportedError);
          }

          // floorDivide & remainder float16 and int8 fallback edge cases
          final fdA = _arr([
            0.0,
            1.0,
            double.nan,
            double.infinity,
            0.0,
            -0.0,
            2.0,
            0.0,
            -0.0,
            -5.0,
            5.9,
          ], DType.float16);
          final fdB = _arr([
            0.0,
            0.0,
            1.0,
            1.0,
            double.infinity,
            double.infinity,
            double.negativeInfinity,
            2.0,
            2.0,
            3.0,
            2.0,
          ], DType.float16);
          expect(floorDivide(fdA, fdB).shape, [11]);
          expect(remainder(fdA, fdB).shape, [11]);
          expect(
            _flat(
              remainder(_arr([-5, 5], DType.int8), _arr([3, -3], DType.int8)),
            ),
            [1, -1],
          );

          // divmod and divmodAs
          final dmX = _arr([7, 8], DType.int32);
          final dmY = _arr([3, 3], DType.int32);
          final badDmOut = NDArray.zeros([3], DType.int32);
          final dispMask = _arr([true, false], DType.boolean)..dispose();
          expect(() => divmod(dmX, dmY, where: dispMask), throwsStateError);
          expect(
            () => divmod(
              _arr([Complex(1, 2)], DType.complex128).asAnySpec,
              _arr([Complex(1, 2)], DType.complex128).asAnySpec,
            ),
            throwsUnsupportedError,
          );
          expect(() => divmod(dmX, dmY, out1: badDmOut), throwsArgumentError);
          expect(() => divmod(dmX, dmY, out2: badDmOut), throwsArgumentError);
          final dmOut1 = NDArray.zeros([2], DType.int32);
          expect(
            () => divmod(dmX, dmY, out1: dmOut1, out2: dmOut1),
            throwsArgumentError,
          );

          // sharesMemory in divmod (with and without where)
          final dmAlias1 = _arr([7, 8], DType.int32);
          divmod(dmAlias1, dmY, out1: dmAlias1, where: posMask);
          final dmAlias2 = _arr([7, 8], DType.int32);
          divmod(dmAlias2, dmY, out1: dmAlias2);
          expect(_flat(dmAlias2), [2, 2]);

          final dispDm = _arr([7, 8], DType.int32)..dispose();
          expect(() => divmodAs(dispDm, dmY, DType.int32), throwsStateError);
          final dmAsSame = divmodAs(dmX, dmY, DType.int32);
          expect(_flat(dmAsSame.quotient), [2, 2]);
          expect(_flat(dmAsSame.remainder), [1, 2]);
          final dmAsMix = divmodAs(
            _arr([7, 8], DType.int16),
            _arr([3, 3], DType.int32),
            DType.int32,
          );
          expect(_flat(dmAsMix.quotient), [2, 2]);
          expect(_flat(dmAsMix.remainder), [1, 2]);
        });
      },
    );

    test('gcd, lcm, heaviside, sign, ceil, floor, round and *As variants', () {
      NDArray.scope(() {
        final a = _arr([12, 18], DType.int32);
        final b = _arr([8, 24], DType.int32);
        final mask = _arr([true, false], DType.boolean);
        final dispMask = _arr([true, false], DType.boolean)..dispose();
        final badOut = NDArray.zeros([3], DType.int32);
        final f64 = _arr([1.0, 2.0], DType.float64);

        expect(() => gcd(a, b, where: dispMask), throwsStateError);
        expect(() => gcd(a, b, out: badOut), throwsArgumentError);
        expect(() => lcm(a, b, where: dispMask), throwsStateError);
        expect(() => lcm(a, b, out: badOut), throwsArgumentError);

        final gcdAlias = _arr([12, 18], DType.int32);
        gcd(gcdAlias, b, out: gcdAlias, where: mask);
        gcd(gcdAlias, b, out: gcdAlias);
        final lcmAlias = _arr([12, 18], DType.int32);
        lcm(lcmAlias, b, out: lcmAlias, where: mask);
        lcm(lcmAlias, b, out: lcmAlias);

        expect(
          () => gcdAs(a, b, DType.int32, where: dispMask),
          throwsStateError,
        );
        expect(
          () => gcdAs(a, b, DType.float64.asAnySpec),
          throwsUnsupportedError,
        );
        expect(_flat(gcdAs(a, b, DType.int32)), [4, 6]);
        expect(_flat(gcdAs(_arr([12, 18], DType.int16), b, DType.int32)), [
          4,
          6,
        ]);
        expect(
          () => lcmAs(a, b, DType.int32, where: dispMask),
          throwsStateError,
        );
        expect(
          () => lcmAs(a, b, DType.float64.asAnySpec),
          throwsUnsupportedError,
        );
        expect(_flat(lcmAs(a, b, DType.int32)), [24, 72]);
        expect(_flat(lcmAs(_arr([12, 18], DType.int16), b, DType.int32)), [
          24,
          72,
        ]);

        // heaviside & heavisideAs
        expect(
          () => heaviside(f64, f64, out: NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );
        final hsF16 = _arr([
          double.nan,
          -1.0,
          0.0,
          1.0,
        ], DType.float16).slice([const Slice(start: 0, stop: 4, step: 2)]);
        final hsF16B = _arr([0.5, 0.5], DType.float16);
        expect(heaviside(hsF16, hsF16B).shape, [2]);
        expect(
          _flat(
            heaviside(_arr([0, 2], DType.uint64), _arr([5, 5], DType.uint64)),
          ),
          [5, 1],
        );
        final hsI16 = _arr([-2, 0, 2], DType.int16);
        expect(_flat(heaviside(hsI16, _arr([5, 5, 5], DType.int16))), [
          0,
          5,
          1,
        ]);
        expect(
          () => heavisideAs(
            _arr([Complex(1, 2)], DType.complex128),
            f64,
            DType.float64,
          ),
          throwsUnsupportedError,
        );

        // sign complex NaN, Infinity, and |real| < |imag|
        final cSign = _arr([
          Complex(double.nan, 1.0),
          Complex(double.infinity, 1.0),
          Complex(1.0, double.infinity),
          Complex(0.0, 0.0),
          Complex(3.0, 4.0),
        ], DType.complex128);
        expect(sign(cSign).shape, [5]);

        // ceil, floor, round out validation and aliasing
        expect(
          () => ceil(f64, out: NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => floor(f64, out: NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => round(f64, out: NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );

        final ceilAlias = _arr([1.2, 2.8], DType.float64);
        ceil(ceilAlias, out: ceilAlias);
        final floorAlias = _arr([1.2, 2.8], DType.float64);
        floor(floorAlias, out: floorAlias);
        final roundAlias = _arr([1.2, 2.8], DType.float64);
        round(roundAlias, out: roundAlias);

        // round decimals == 0 (float16), decimals < 0, decimals > 0, huge decimals
        final rF16 = _arr([
          double.nan,
          double.infinity,
          0.0,
          1.2,
          1.8,
          2.5,
          3.5,
        ], DType.float16);
        expect(round(rF16, decimals: 0).shape, [7]);

        final rBool = _arr([true, false], DType.boolean);
        expect(_flat(round(rBool, decimals: -1)), [false, false]);
        expect(_flat(round(_arr([123, 456], DType.int32), decimals: -1)), [
          120,
          460,
        ]);
        expect(_flat(round(_arr([123, 456], DType.int32), decimals: -400)), [
          0,
          0,
        ]);
        expect(_flat(round(_arr([123, 456], DType.uint64), decimals: -1)), [
          120,
          460,
        ]);
        expect(
          () => round(_arr([Complex(12.3, 45.6)], DType.complex128)),
          throwsUnsupportedError,
        );

        final rSpecial = _arr([
          double.nan,
          double.infinity,
          0.0,
          1.234,
          1e308,
        ], DType.float64);
        expect(round(rSpecial, decimals: 2).shape, [5]);
        expect(round(rSpecial, decimals: 400).shape, [5]);
        expect(round(rSpecial, decimals: -400).shape, [5]);
        expect(round(_arr([-1.2], DType.float64), decimals: -400).shape, [1]);
      });
    });

    test(
      'power across all 15 dtypes, add/sub/mul/div same-dtype non-kernel dtypes, out aliasing, and *As variants',
      () {
        NDArray.scope(() {
          final allDTypes = <DType<DTypeTag>>[
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
            DType.boolean,
            DType.complex128,
            DType.complex64,
          ];
          for (final dt in allDTypes) {
            final ca = _makeSample(dt).asAnySpec;
            final cb = _makeSample(dt).asAnySpec;
            final sa = _makeSample(dt, strided: true).asAnySpec;
            final sb = _makeSample(dt, strided: true).asAnySpec;
            expect(power(ca, cb).shape, [4]);
            expect(power(sa, sb).shape, [2]);
          }
          for (final dt in [
            DType.uint64,
            DType.uint32,
            DType.uint16,
            DType.int8,
            DType.float16,
            DType.bfloat16,
            DType.boolean,
          ]) {
            final ca = _makeSample(dt).asAnySpec;
            final cb = _makeSample(dt).asAnySpec;
            final sa = _makeSample(dt, strided: true).asAnySpec;
            final sb = _makeSample(dt, strided: true).asAnySpec;
            final mask4 = _arr([true, false, true, false], DType.boolean);
            expect(add(ca, cb, where: mask4).shape, [4]);
            expect(add(sa, sb).shape, [2]);
            if (dt != DType.boolean) {
              expect(subtract(ca, cb, where: mask4).shape, [4]);
              expect(subtract(sa, sb).shape, [2]);
            }
            expect(multiply(ca, cb, where: mask4).shape, [4]);
            expect(multiply(sa, sb).shape, [2]);
            expect(divide(ca, cb, where: mask4).shape, [4]);
            expect(divide(sa, sb).shape, [2]);
          }

          // out aliasing (with and without where) across unary and binary ops
          final f64 = _arr([1.0, 2.0], DType.float64);
          final mask2 = _arr([true, false], DType.boolean);
          for (final w in <NDArray<Boolean>?>[null, mask2]) {
            sqrt(f64, where: w, out: f64);
            expm1(f64, where: w, out: f64);
            log1p(f64, where: w, out: f64);
            logaddexp(f64, f64, where: w, out: f64);
            logaddexp2(f64, f64, where: w, out: f64);
            rint(f64, where: w, out: f64);
            trunc(f64, where: w, out: f64);
            square(f64, where: w, out: f64);
            reciprocal(f64, where: w, out: f64);
            power(f64, f64, where: w, out: f64);
            negative(f64, where: w, out: f64);
            abs(f64, where: w, out: f64);
            floorDivide(f64, f64, where: w, out: f64);
            remainder(f64, f64, where: w, out: f64);
            fmod(f64, f64, where: w, out: f64);
            heaviside(f64, f64, where: w, out: f64);
            sign(f64, where: w, out: f64);
            ceil(f64, where: w, out: f64);
            floor(f64, where: w, out: f64);
            round(f64, where: w, out: f64);
            add(f64, f64, where: w, out: f64);
            subtract(f64, f64, where: w, out: f64);
            multiply(f64, f64, where: w, out: f64);
            divide(f64, f64, where: w, out: f64);
          }
          final bOut = _arr([true, false], DType.boolean);
          add(bOut, bOut, out: bOut);
          multiply(bOut, bOut, out: bOut);
          expect(_flat(square(bOut)), [true, false]);
          expect(_flat(sqrt(bOut)), [1.0, 0.0]);
          expect(_flat(sign(bOut)), [true, false]);
          expect(() => abs(bOut.asAnySpec), throwsUnsupportedError);
          final outF16 = NDArray.zeros([2], DType.float16);
          logaddexpAs(f64, f64, DType.float16, out: outF16);
          logaddexp2As(f64, f64, DType.float16, out: outF16);
          expect(logaddexpAs(f64, f64, DType.float16).dtype, DType.float16);
          expect(logaddexp2As(f64, f64, DType.float16).dtype, DType.float16);

          // floorDivideAs, remainderAs, modAs, fmodAs, floatPower, floatPowerAs, heavisideAs
          final i32 = _arr([7, 8], DType.int32);
          final i16 = _arr([3, 3], DType.int16);
          final i32b = _arr([3, 3], DType.int32);
          final disp = _arr([1, 2], DType.int32)..dispose();

          expect(() => floorDivideAs(disp, i32, DType.int32), throwsStateError);
          expect(_flat(floorDivideAs(i32, i32b, DType.int32)), [2, 2]);
          expect(_flat(floorDivideAs(i32, i16, DType.int32)), [2, 2]);

          expect(() => remainderAs(disp, i32, DType.int32), throwsStateError);
          expect(_flat(remainderAs(i32, i32b, DType.int32)), [1, 2]);
          expect(_flat(remainderAs(i32, i16, DType.int32)), [1, 2]);
          expect(_flat(modAs(i32, i16, DType.int32)), [1, 2]);

          expect(() => fmodAs(disp, i32, DType.int32), throwsStateError);
          expect(_flat(fmodAs(i32, i32b, DType.int32)), [1, 2]);
          expect(_flat(fmodAs(i32, i16, DType.int32)), [1, 2]);

          expect(() => floatPower(disp, i32), throwsStateError);
          expect(
            () => floatPower(i32.asAnySpec, i16.asAnySpec),
            throwsArgumentError,
          );
          expect(_flat(floatPower(i32, i32b)), [343.0, 512.0]);
          final cpx = _arr([Complex(2, 0), Complex(3, 0)], DType.complex64);
          expect(floatPower(cpx, cpx).dtype, DType.complex128);

          expect(
            () => floatPowerAs(disp, i32, DType.float64),
            throwsStateError,
          );
          expect(_flat(floatPowerAs(i32, i16, DType.float64)), [343.0, 512.0]);
          expect(_flat(floatPowerAs(i32, i16, DType.int32)), [343, 512]);
          final outI32 = NDArray.zeros([2], DType.int32);
          floatPowerAs(i32, i16, DType.int32, out: outI32);
          expect(_flat(outI32), [343, 512]);
          expect(
            () => floatPowerAs(
              i32,
              i16,
              DType.int32,
              out: NDArray.zeros([3], DType.int32),
            ),
            throwsArgumentError,
          );

          expect(() => heavisideAs(disp, i32, DType.float64), throwsStateError);
          expect(_flat(heavisideAs(i32, i32b, DType.int32)), [1, 1]);
          expect(_flat(heavisideAs(i32, i16, DType.float64)), [1.0, 1.0]);
        });
      },
    );
  });
}
