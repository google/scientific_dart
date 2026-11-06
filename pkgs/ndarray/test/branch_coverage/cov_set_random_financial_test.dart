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

void main() {
  group('set_operations branch coverage', () {
    test('unique variants validation, empty, and small arrays', () {
      NDArray.scope(() {
        final disposed = NDArray.fromList([1, 2], [2], DType.int32)..dispose();
        final valid = NDArray.fromList([3, 1, 2, 1], [4], DType.int32);
        final outDisposed = NDArray.zeros([3], DType.int32)..dispose();

        expect(() => unique(disposed), throwsStateError);
        expect(() => unique(valid, out: outDisposed), throwsStateError);
        expect(
          () => unique<DTypeTag, DTypeTag>(
            valid,
            out: NDArray.zeros([3], DType.float64),
          ),
          throwsArgumentError,
        );

        // Empty array across unique variants
        final empty = NDArray.create([0], DType.float64);
        final outEmpty = NDArray.create([0], DType.float64);
        final outBad = NDArray.create([1], DType.float64);
        expect(unique(empty).size, 0);
        expect(unique(empty, out: outEmpty).size, 0);
        expect(() => unique(empty, out: outBad), throwsArgumentError);

        final emptyAll = uniqueAll(empty);
        expect(emptyAll.values.size, 0);
        expect(emptyAll.index.size, 0);
        expect(emptyAll.inverse.size, 0);
        expect(emptyAll.counts.size, 0);

        // Non-1D and non-table path (float64) with out
        final mat = NDArray.fromList(
          [3.0, 1.0, 2.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final outF64 = NDArray.zeros([3], DType.float64);
        expect(unique(mat, out: outF64).toList(), [1.0, 2.0, 3.0]);
        expect(() => unique(mat, out: outBad), throwsArgumentError);

        final withIdx = uniqueWithIndex(mat);
        expect(withIdx.values.toList(), [1.0, 2.0, 3.0]);
        expect(withIdx.index.toList(), [1, 2, 0]);

        final withInv = uniqueWithInverse(mat);
        expect(withInv.values.toList(), [1.0, 2.0, 3.0]);
        expect(withInv.inverse.toList(), [2, 0, 1, 0]);

        final withCnt = uniqueWithCounts(mat);
        expect(withCnt.values.toList(), [1.0, 2.0, 3.0]);
        expect(withCnt.counts.toList(), [2, 1, 1]);

        final allRes = uniqueAll(mat);
        expect(allRes.values.toList(), [1.0, 2.0, 3.0]);
        expect(allRes.counts.toList(), [2, 1, 1]);
      });
    });

    test('unique table path across all 7 integer dtypes (size > 64)', () {
      NDArray.scope(() {
        // Values have decreasing and increasing elements to hit v < minVal and v > maxVal
        final raw = <int>[
          50,
          10,
          90,
          10,
          50,
          for (var i = 0; i < 65; i++) 20 + (i % 5),
        ];
        final expectedUniqueCount = 8; // 10, 20, 21, 22, 23, 24, 50, 90

        void checkDType<T extends DTypeTag>(DType<T> dtype) {
          final arr = NDArray<T>.fromList(raw, [raw.length], dtype);
          final u = unique(arr);
          expect(u.size, expectedUniqueCount);

          final out = NDArray<T>.create([expectedUniqueCount], dtype);
          unique(arr, out: out);
          expect(out.size, expectedUniqueCount);

          final outBase = NDArray<T>.create([expectedUniqueCount * 2], dtype);
          final outStrided = outBase.slice([Slice(step: 2)]);
          unique(arr, out: outStrided);
          expect(outStrided.size, expectedUniqueCount);

          final outBad = NDArray<T>.create([1], dtype);
          expect(() => unique(arr, out: outBad), throwsArgumentError);

          final uc = uniqueWithCounts(arr);
          expect(uc.values.size, expectedUniqueCount);
          expect(uc.counts.size, expectedUniqueCount);

          final outCnt = NDArray<T>.create([expectedUniqueCount], dtype);
          uniqueWithCounts(arr, out: outCnt);
          expect(outCnt.size, expectedUniqueCount);

          final outCntStrided = outBase.slice([Slice(step: 2)]);
          uniqueWithCounts(arr, out: outCntStrided);
          expect(outCntStrided.size, expectedUniqueCount);

          expect(() => uniqueWithCounts(arr, out: outBad), throwsArgumentError);
        }

        checkDType(DType.int32);
        checkDType(DType.int64);
        checkDType(DType.int16);
        checkDType(DType.int8);
        checkDType(DType.uint32);
        checkDType(DType.uint16);
        checkDType(DType.uint8);

        // Table fallback branches:
        // 1. span > maxAllowedSpan (262144) but <= maxSpan (16777216)
        final mediumSpan = NDArray.fromList(
          [0, 300000, for (var i = 0; i < 68; i++) 1],
          [70],
          DType.int32,
        );
        expect(unique(mediumSpan).toList(), [0, 1, 300000]);

        // 2. span > maxSpan (16777216)
        final hugeSpan = NDArray.fromList(
          [0, 20000000, for (var i = 0; i < 68; i++) 1],
          [70],
          DType.int32,
        );
        expect(unique(hugeSpan).toList(), [0, 1, 20000000]);

        // 3. int64 span < 0 overflow
        final overflowSpan = NDArray.fromList(
          [-10, 0x7fffffffffffffff, for (var i = 0; i < 68; i++) 1],
          [70],
          DType.int64,
        );
        expect(unique(overflowSpan).size, 3);

        // 4. values.size * 16 > 262144 (size > 16384)
        final largeArr = NDArray<Int32>.zeros([16500], DType.int32);
        expect(unique(largeArr).toList(), [0]);

        // 5. Non-integer dtype with size > 64 (hits _minMaxInt returning null)
        final largeF64 = NDArray<Float64>.zeros([70], DType.float64);
        expect(unique(largeF64).toList(), [0.0]);
      });
    });

    test('intersect1d, setdiff1d, setxor1d, union1d branches', () {
      NDArray.scope(() {
        final disposed = NDArray.fromList([1, 2], [2], DType.int32)..dispose();
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final b = NDArray.fromList([2, 3, 4], [3], DType.int32);
        final outDisposed = NDArray.zeros([2], DType.int32)..dispose();

        // Disposed checks
        for (final op in [intersect1d, setdiff1d, setxor1d, union1d]) {
          expect(() => op(disposed, b), throwsStateError);
          expect(() => op(a, disposed), throwsStateError);
          expect(() => op(a, b, out: outDisposed), throwsStateError);
          expect(
            () => op<DTypeTag, DTypeTag>(
              a,
              b,
              out: NDArray.zeros([2], DType.float64),
            ),
            throwsArgumentError,
          );
        }

        // Mixed dtypes & 2D inputs & assumeUnique
        final m1 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final m2 = NDArray.fromList(
          [3.0, 4.0, 5.0, 6.0],
          [2, 2],
          DType.float64,
        );
        expect(
          intersect1d<DTypeTag, DTypeTag>(m1, m2, assumeUnique: true).toList(),
          [3.0, 4.0],
        );
        expect(
          setdiff1d<DTypeTag, DTypeTag>(m1, m2, assumeUnique: true).toList(),
          [1.0, 2.0],
        );
        expect(
          setxor1d<DTypeTag, DTypeTag>(m1, m2, assumeUnique: true).toList(),
          [1.0, 2.0, 5.0, 6.0],
        );
        expect(union1d<DTypeTag, DTypeTag>(m1, m2).toList(), [
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
        ]);

        // Empty inputs (maxDstSize == 0)
        final empty = NDArray.create([0], DType.int32);
        final out0 = NDArray.create([0], DType.int32);
        final out1 = NDArray.create([1], DType.int32);
        final out2 = NDArray.create([2], DType.int32);
        final out4 = NDArray.create([4], DType.int32);

        expect(intersect1d(empty, a).size, 0);
        expect(intersect1d(empty, a, out: out0).size, 0);
        expect(() => intersect1d(empty, a, out: out1), throwsArgumentError);

        expect(setdiff1d(empty, a).size, 0);
        expect(setdiff1d(empty, a, out: out0).size, 0);
        expect(() => setdiff1d(empty, a, out: out1), throwsArgumentError);

        expect(setxor1d(empty, empty).size, 0);
        expect(setxor1d(empty, empty, out: out0).size, 0);
        expect(() => setxor1d(empty, empty, out: out1), throwsArgumentError);

        expect(union1d(empty, empty).size, 0);
        expect(union1d(empty, empty, out: out0).size, 0);
        expect(() => union1d(empty, empty, out: out1), throwsArgumentError);

        // Non-empty inputs producing 0 count
        final c1 = NDArray.fromList([1, 2], [2], DType.int32);
        final c2 = NDArray.fromList([3, 4], [2], DType.int32);
        expect(intersect1d(c1, c2).size, 0);
        expect(intersect1d(c1, c2, out: out0).size, 0);
        expect(() => intersect1d(c1, c2, out: out1), throwsArgumentError);

        expect(setdiff1d(c1, c1).size, 0);
        expect(setdiff1d(c1, c1, out: out0).size, 0);
        expect(() => setdiff1d(c1, c1, out: out1), throwsArgumentError);

        expect(setxor1d(c1, c1).size, 0);
        expect(setxor1d(c1, c1, out: out0).size, 0);
        expect(() => setxor1d(c1, c1, out: out1), throwsArgumentError);

        // Non-empty results with out & bad out shape
        expect(intersect1d(a, b, out: out2).toList(), [2, 3]);
        expect(() => intersect1d(a, b, out: out1), throwsArgumentError);

        expect(setdiff1d(a, b, out: out1).toList(), [1]);
        expect(() => setdiff1d(a, b, out: out2), throwsArgumentError);

        expect(setxor1d(a, b, out: out2).toList(), [1, 4]);
        expect(() => setxor1d(a, b, out: out1), throwsArgumentError);

        expect(union1d(a, b, out: out4).toList(), [1, 2, 3, 4]);
        expect(() => union1d(a, b, out: out1), throwsArgumentError);
      });
    });

    test('isin across all dtypes, table paths, fallbacks, and out aliasing', () {
      NDArray.scope(() {
        final disposed = NDArray.fromList([1, 2], [2], DType.int32)..dispose();
        final valid = NDArray.fromList([1, 2], [2], DType.int32);
        final outDisposed = NDArray.zeros([2], DType.boolean)..dispose();

        expect(() => isin(disposed, valid), throwsStateError);
        expect(() => isin(valid, disposed), throwsStateError);
        expect(() => isin(valid, valid, out: outDisposed), throwsStateError);
        expect(
          () => isin(valid, valid, out: NDArray.zeros([3], DType.boolean)),
          throwsArgumentError,
        );

        // Empty element and empty testElements
        final empty = NDArray.create([0], DType.int32);
        expect(isin(empty, valid).size, 0);
        expect(isin(valid, empty, invert: false).toList(), [false, false]);
        expect(isin(valid, empty, invert: true).toList(), [true, true]);

        // Mixed dtypes & non-contiguous element/testElements/out
        final elemMat = NDArray.fromList(
          [1, 2, 3, 4],
          [2, 2],
          DType.int32,
        ).transpose();
        final testMat = NDArray.fromList([2.0, 4.0], [1, 2], DType.float64);
        final outBase = NDArray.zeros([2, 4], DType.boolean);
        final outStrided = outBase.slice([Slice.all(), Slice(step: 2)]);
        isin<DTypeTag>(elemMat, testMat, out: outStrided);
        expect(outStrided.toList(), [false, false, true, true]);

        // Table path across all 7 integer dtypes (invert: false and invert: true)
        void checkIntIsin<T extends DTypeTag>(DType<T> dtype) {
          // Elements cover: < minVal (1), in-range present (10), in-range absent (15), > maxVal (30)
          final elem = NDArray<T>.fromList([1, 10, 15, 30], [4], dtype);
          // Test elements have decreasing and increasing order to cover v < minVal and v > maxVal
          final testArr = NDArray<T>.fromList([10, 5, 20], [3], dtype);
          expect(isin(elem, testArr, invert: false).toList(), [
            false,
            true,
            false,
            false,
          ]);
          expect(isin(elem, testArr, invert: true).toList(), [
            true,
            false,
            true,
            true,
          ]);
        }

        checkIntIsin(DType.int32);
        checkIntIsin(DType.int64);
        checkIntIsin(DType.int16);
        checkIntIsin(DType.int8);
        checkIntIsin(DType.uint8);
        checkIntIsin(DType.uint16);
        checkIntIsin(DType.uint32);

        // Boolean table path: both true & false, only true, only false, and aliasing out
        final bElem = NDArray.fromList([true, false], [2], DType.boolean);
        final bBoth = NDArray.fromList([false, true], [2], DType.boolean);
        final bOnlyTrue = NDArray.fromList([true, true], [2], DType.boolean);
        final bOnlyFalse = NDArray.fromList([false, false], [2], DType.boolean);

        expect(isin(bElem, bBoth, invert: false).toList(), [true, true]);
        expect(isin(bElem, bBoth, invert: true).toList(), [false, false]);
        expect(isin(bElem, bOnlyTrue, invert: false).toList(), [true, false]);
        expect(isin(bElem, bOnlyTrue, invert: true).toList(), [false, true]);
        expect(isin(bElem, bOnlyFalse, invert: false).toList(), [false, true]);
        expect(isin(bElem, bOnlyFalse, invert: true).toList(), [true, false]);

        // Out aliasing element and testElements
        final bAlias1 = NDArray.fromList([true, false], [2], DType.boolean);
        isin(bAlias1, bOnlyTrue, out: bAlias1);
        expect(bAlias1.toList(), [true, false]);

        final bAlias2 = NDArray.fromList([true, false], [2], DType.boolean);
        isin(bElem, bAlias2, out: bAlias2);
        expect(bAlias2.toList(), [true, true]);

        // Table fallback to ndarray_isin (range > maxTableRange and float64)
        final bigInt32 = NDArray.fromList([0, 20000000], [2], DType.int32);
        expect(
          isin(bigInt32, bigInt32, assumeUnique: true, invert: false).toList(),
          [true, true],
        );
        final bigUint32 = NDArray.fromList([0, 20000000], [2], DType.uint32);
        expect(
          isin(
            bigUint32,
            bigUint32,
            assumeUnique: false,
            invert: true,
          ).toList(),
          [false, false],
        );
        final bigInt64 = NDArray.fromList([0, 20000000], [2], DType.int64);
        expect(isin(bigInt64, bigInt64).toList(), [true, true]);
        final overflowInt64 = NDArray.fromList(
          [-10, 0x7fffffffffffffff],
          [2],
          DType.int64,
        );
        expect(isin(overflowInt64, overflowInt64).toList(), [true, true]);
      });
    });
  });

  group('random branch coverage', () {
    test(
      'RandomGenerator and basic distributions with secure, dtypes, and out',
      () {
        NDArray.scope(() {
          final rng = RandomGenerator(42);
          final rngDefault = RandomGenerator();
          expect(rngDefault.uniform([2]).size, 2);
          expect(rng.uniform([2], secure: true).size, 2);
          expect(rng.randint([2], low: 0, high: 10, secure: true).size, 2);
          expect(rng.normal([2], secure: true).size, 2);
          expect(rng.exponential([2], secure: true).size, 2);

          final disposedOut = NDArray.zeros([2], DType.float64)..dispose();
          expect(() => uniform([2], out: disposedOut), throwsStateError);
          expect(
            () => uniform([2], out: NDArray.zeros([3], DType.float64)),
            throwsArgumentError,
          );
          expect(
            () => uniform([2], secure: true, seed: 1),
            throwsArgumentError,
          );
          expect(
            () => uniform([2], dtype: DType.int32 as DType<AnySpec>),
            throwsArgumentError,
          );

          // uniform float64 & float32 with contiguous and non-contiguous out
          for (final sec in [false, true]) {
            final u64Base = NDArray.zeros([4], DType.float64);
            uniform(
              [2],
              dtype: DType.float64,
              secure: sec,
              out: u64Base.slice([Slice(step: 2)]),
            );
            final u32Base = NDArray.zeros([4], DType.float32);
            uniform(
              [2],
              dtype: DType.float32,
              secure: sec,
              out: u32Base.slice([Slice(step: 2)]),
            );
            uniform([2], dtype: DType.float32, secure: sec);
          }

          // randint validation & dtypes
          expect(() => randint([2], low: 5, high: 5), throwsArgumentError);
          expect(
            () => randint([2], low: 0, high: 5, secure: true, seed: 1),
            throwsArgumentError,
          );
          expect(
            () => randint(
              [2],
              low: 0,
              high: 5,
              dtype: DType.float64 as DType<AnySpec>,
            ),
            throwsArgumentError,
          );
          expect(
            () => randint([2], low: -2147483649, high: 0, dtype: DType.int32),
            throwsArgumentError,
          );
          expect(
            () => randint([2], low: 0, high: 2147483649, dtype: DType.int32),
            throwsArgumentError,
          );
          expect(
            () => randint([2], low: -32769, high: 0, dtype: DType.int16),
            throwsArgumentError,
          );
          expect(
            () => randint([2], low: 0, high: 32769, dtype: DType.int16),
            throwsArgumentError,
          );
          expect(
            () => randint([2], low: -1, high: 10, dtype: DType.uint8),
            throwsArgumentError,
          );
          expect(
            () => randint([2], low: 0, high: 257, dtype: DType.uint8),
            throwsArgumentError,
          );

          for (final sec in [false, true]) {
            for (final dt in <DType<AnySpec>>[
              DType.int64,
              DType.int32,
              DType.int16,
              DType.uint8,
            ]) {
              randint([2], low: 0, high: 10, dtype: dt, secure: sec);
              final base = NDArray.zeros([4], dt);
              randint(
                [2],
                low: 0,
                high: 10,
                dtype: dt,
                secure: sec,
                out: base.slice([Slice(step: 2)]),
              );
            }
          }

          // normal validation & dtypes
          expect(() => normal([2], scale: 0.0), throwsArgumentError);
          expect(() => normal([2], secure: true, seed: 1), throwsArgumentError);
          expect(
            () => normal([2], dtype: DType.int32 as DType<AnySpec>),
            throwsArgumentError,
          );
          for (final sec in [false, true]) {
            for (final dt in <DType<AnySpec>>[DType.float64, DType.float32]) {
              normal([2], dtype: dt, secure: sec);
              final base = NDArray.zeros([4], dt);
              normal(
                [2],
                dtype: dt,
                secure: sec,
                out: base.slice([Slice(step: 2)]),
              );
            }
          }

          // exponential validation & dtypes
          expect(() => exponential([2], scale: 0.0), throwsArgumentError);
          expect(() => exponential([2], lam: -1.0), throwsArgumentError);
          expect(
            () => exponential([2], secure: true, seed: 1),
            throwsArgumentError,
          );
          expect(
            () => exponential([2], dtype: DType.int32 as DType<AnySpec>),
            throwsArgumentError,
          );
          for (final sec in [false, true]) {
            for (final dt in <DType<AnySpec>>[DType.float64, DType.float32]) {
              exponential([2], lam: 2.0, dtype: dt, secure: sec);
              final base = NDArray.zeros([4], dt);
              exponential(
                [2],
                dtype: dt,
                secure: sec,
                out: base.slice([Slice(step: 2)]),
              );
            }
          }

          // poisson & binomial
          final disposedI64 = NDArray.zeros([2], DType.int64)..dispose();
          expect(() => poisson([2], out: disposedI64), throwsStateError);
          expect(
            () => poisson([2], secure: true, seed: 1),
            throwsArgumentError,
          );
          expect(() => poisson([2], lam: 0.0), throwsArgumentError);
          expect(
            () => poisson([2], dtype: DType.float64 as DType<AnySpec>),
            throwsArgumentError,
          );

          expect(
            () => binomial([2], n: 5, p: 0.5, out: disposedI64),
            throwsStateError,
          );
          expect(
            () => binomial([2], n: 5, p: 0.5, secure: true, seed: 1),
            throwsArgumentError,
          );
          expect(() => binomial([2], n: -1, p: 0.5), throwsArgumentError);
          expect(() => binomial([2], n: 5, p: -0.1), throwsArgumentError);
          expect(() => binomial([2], n: 5, p: 1.1), throwsArgumentError);
          expect(() => binomial([2], n: 5, p: double.nan), throwsArgumentError);
          expect(
            () => binomial(
              [2],
              n: 5,
              p: 0.5,
              dtype: DType.float64 as DType<AnySpec>,
            ),
            throwsArgumentError,
          );

          for (final sec in [false, true]) {
            for (final dt in <DType<AnySpec>>[DType.int64, DType.int32]) {
              poisson([2], lam: 4.0, dtype: dt, secure: sec);
              final pBase = NDArray.zeros([4], dt);
              poisson(
                [2],
                lam: 4.0,
                dtype: dt,
                secure: sec,
                out: pBase.slice([Slice(step: 2)]),
              );

              binomial([2], n: 10, p: 0.5, dtype: dt, secure: sec);
              final bBase = NDArray.zeros([4], dt);
              binomial(
                [2],
                n: 10,
                p: 0.5,
                dtype: dt,
                secure: sec,
                out: bBase.slice([Slice(step: 2)]),
              );
            }
          }
        });
      },
    );

    test('multivariateNormal, multinomial, choice, shuffle, permutation', () {
      NDArray.scope(() {
        final mean = NDArray.fromList([0.0, 0.0], [2], DType.float64);
        final cov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final disposed = NDArray.fromList([0.0, 0.0], [2], DType.float64)
          ..dispose();
        final disposedOut = NDArray.zeros([2], DType.float64)..dispose();

        expect(() => multivariateNormal(disposed, cov), throwsStateError);
        expect(() => multivariateNormal(mean, disposed), throwsStateError);
        expect(
          () => multivariateNormal(mean, cov, out: disposedOut),
          throwsStateError,
        );
        expect(
          () => multivariateNormal(mean, cov, secure: true, seed: 1),
          throwsArgumentError,
        );
        expect(
          () => multivariateNormal<SelfOf<DTypeTag>>(
            mean,
            NDArray.zeros([2, 2], DType.float32),
          ),
          throwsArgumentError,
        );
        expect(() => multivariateNormal(cov, cov), throwsArgumentError);
        expect(() => multivariateNormal(mean, mean), throwsArgumentError);
        expect(
          () => multivariateNormal(mean, NDArray.zeros([3, 3], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => multivariateNormal(
            NDArray.fromList([1, 2], [2], DType.int32),
            NDArray.fromList([1, 0, 0, 1], [2, 2], DType.int32),
          ),
          throwsArgumentError,
        );
        // Negative diagonal, non-symmetric, and indefinite PSD fallback error
        expect(
          () => multivariateNormal(
            mean,
            NDArray.fromList([-1.0, 0.0, 0.0, 1.0], [2, 2], DType.float64),
          ),
          throwsArgumentError,
        );
        expect(
          () => multivariateNormal(
            mean,
            NDArray.fromList([1.0, 2.0, 3.0, 1.0], [2, 2], DType.float64),
          ),
          throwsArgumentError,
        );
        expect(
          () => multivariateNormal(
            mean,
            NDArray.fromList([1.0, 2.0, 2.0, 1.0], [2, 2], DType.float64),
          ),
          throwsArgumentError,
        );

        // d == 0
        expect(
          multivariateNormal(
            NDArray.create([0], DType.float64),
            NDArray.create([0, 0], DType.float64),
          ).size,
          0,
        );

        // Cast mean/cov to float32 & PSD singular matrix fallback in float32 and float64
        final psdCov64 = NDArray.fromList(
          [1.0, 1.0, 1.0, 1.0],
          [2, 2],
          DType.float64,
        );
        expect(
          multivariateNormal<AnySpec>(
            mean,
            psdCov64,
            size: [2],
            dtype: DType.float32,
            seed: 42,
          ).shape,
          [2, 2],
        );
        expect(multivariateNormal(mean, psdCov64, size: [2], seed: 42).shape, [
          2,
          2,
        ]);

        // Out aliasing mean / cov / non-contiguous
        final aliasMean = NDArray.fromList([0.0, 0.0], [2], DType.float64);
        multivariateNormal(aliasMean, cov, seed: 1, out: aliasMean);
        final aliasCov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );
        multivariateNormal(mean, aliasCov, size: [2], seed: 1, out: aliasCov);

        // multinomial
        final pvals = NDArray.fromList([0.5, 0.5], [2], DType.float64);
        final disposedI64 = NDArray.zeros([2], DType.int64)..dispose();
        expect(() => multinomial(5, disposed), throwsStateError);
        expect(() => multinomial(5, pvals, out: disposedI64), throwsStateError);
        expect(
          () => multinomial(5, pvals, secure: true, seed: 1),
          throwsArgumentError,
        );
        expect(() => multinomial(-1, pvals), throwsArgumentError);
        expect(
          () => multinomial(
            5,
            NDArray.fromList([true, false], [2], DType.boolean),
          ),
          throwsArgumentError,
        );
        expect(() => multinomial(5, cov), throwsArgumentError);
        expect(
          () => multinomial(5, NDArray.create([0], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => multinomial(5, pvals, dtype: DType.float64 as DType<AnySpec>),
          throwsArgumentError,
        );
        // uint64 pvals with negative signed int (large unsigned value) and valid [1, 0]
        final u64Bad = NDArray.fromList([-1, 1], [2], DType.uint64);
        expect(() => multinomial(5, u64Bad), throwsArgumentError);
        final u64Good = NDArray.fromList([1, 0], [2], DType.uint64);
        expect(multinomial(5, u64Good, seed: 42).toList(), [5, 0]);
        expect(
          () =>
              multinomial(5, NDArray.fromList([0.0, 0.0], [2], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () =>
              multinomial(5, NDArray.fromList([0.2, 0.3], [2], DType.float64)),
          throwsArgumentError,
        );

        // multinomial int32, int64, secure, and non-contiguous out
        expect(
          multinomial(
            5,
            pvals,
            size: [2],
            dtype: DType.int32,
            secure: true,
          ).shape,
          [2, 2],
        );
        final mOutBase = NDArray.zeros([2, 4], DType.int64);
        final mOutStrided = mOutBase.slice([Slice.all(), Slice(step: 2)]);
        multinomial(5, pvals, size: [2], seed: 42, out: mOutStrided);
        expect(mOutStrided.shape, [2, 2]);

        // choice
        final arr = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
        final probs = NDArray.fromList([0.2, 0.3, 0.5], [3], DType.float64);
        expect(() => choice(disposed), throwsStateError);
        expect(() => choice(arr, p: disposed), throwsStateError);
        expect(
          () => choice(arr, size: [2], out: disposedOut),
          throwsStateError,
        );
        expect(() => choice(arr, secure: true, seed: 1), throwsArgumentError);
        expect(() => choice(cov), throwsArgumentError);
        expect(() => choice(arr, p: cov), throwsArgumentError);
        expect(() => choice(arr, p: pvals), throwsArgumentError);
        expect(
          () => choice(
            arr,
            p: NDArray.fromList([-0.1, 0.6, 0.5], [3], DType.float64),
          ),
          throwsArgumentError,
        );
        expect(
          () => choice(
            arr,
            p: NDArray.fromList([0.1, 0.2, 0.3], [3], DType.float64),
          ),
          throwsArgumentError,
        );
        final emptyArr = NDArray.create([0], DType.float64);
        expect(choice(emptyArr, size: [0]).size, 0);
        expect(() => choice(emptyArr, size: [1]), throwsArgumentError);
        expect(
          () => choice(arr, size: [4], replace: false),
          throwsArgumentError,
        );
        expect(choice(arr, size: [0]).size, 0);

        for (final sec in [false, true]) {
          for (final rep in [true, false]) {
            choice(arr, size: [2], replace: rep, secure: sec);
            choice(arr, size: [2], replace: rep, p: probs, secure: sec);
          }
        }
        // choice out aliasing p
        final aliasP = NDArray.fromList([0.5, 0.5], [2], DType.float64);
        final twoElem = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        choice(twoElem, size: [2], p: aliasP, seed: 1, out: aliasP);

        // shuffle & permutation
        expect(() => shuffle(disposed), throwsStateError);
        expect(() => shuffle(arr, secure: true, seed: 1), throwsArgumentError);
        final scalar = NDArray.scalar(1.0, dtype: DType.float64);
        shuffle(scalar);
        final oneElem = NDArray.fromList([1.0], [1], DType.float64);
        shuffle(oneElem);
        final s1d = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        shuffle(s1d, secure: true);
        final s2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        shuffle(s2d, secure: true);
        shuffle(s2d, seed: 42);

        expect(() => permutation(disposed), throwsStateError);
        expect(() => permutation(arr, out: disposedOut), throwsStateError);
        expect(
          () => permutation(arr, secure: true, seed: 1),
          throwsArgumentError,
        );
        final permAlias = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        permutation(permAlias, seed: 42, out: permAlias);
      });
    });
  });

  group('financial branch coverage', () {
    test('fv, pv, pmt across scalar, 1D, strided 2D, and out aliasing', () {
      NDArray.scope(() {
        final disposed = NDArray.scalar(0.05, dtype: DType.float64)..dispose();
        final r = NDArray.scalar(0.05, dtype: DType.float64);
        final n = NDArray.scalar(10.0, dtype: DType.float64);
        final p = NDArray.scalar(-100.0, dtype: DType.float64);
        final v = NDArray.scalar(-1000.0, dtype: DType.float64);
        final f = NDArray.scalar(2000.0, dtype: DType.float64);

        expect(() => fv(disposed, n, p, v), throwsStateError);
        expect(() => fv(r, n, p, v, out: disposed), throwsStateError);
        expect(() => pv(disposed, n, p, f), throwsStateError);
        expect(() => pv(r, n, p, f, out: disposed), throwsStateError);
        expect(() => pmt(disposed, n, v), throwsStateError);
        expect(() => pmt(r, n, v, fv: disposed), throwsStateError);
        expect(() => pmt(r, n, v, out: disposed), throwsStateError);

        final badOut = NDArray.zeros([5], DType.float64);
        expect(() => fv(r, n, p, v, out: badOut), throwsArgumentError);

        // Empty array
        final empty = NDArray.create([0], DType.float64);
        expect(fv(empty, empty, empty, empty).size, 0);

        // Out aliasing pv and fv, and contiguous out
        final outAliasPv = NDArray.fromList([-1000.0], [1], DType.float64);
        fv(
          NDArray.fromList([0.05], [1], DType.float64),
          NDArray.fromList([10.0], [1], DType.float64),
          NDArray.fromList([-100.0], [1], DType.float64),
          outAliasPv,
          out: outAliasPv,
        );
        final outAliasFv = NDArray.fromList([2000.0], [1], DType.float64);
        pv(
          NDArray.fromList([0.05], [1], DType.float64),
          NDArray.fromList([10.0], [1], DType.float64),
          NDArray.fromList([-100.0], [1], DType.float64),
          outAliasFv,
          out: outAliasFv,
        );
        final outContig = NDArray.zeros([1], DType.float64);
        pmt(
          NDArray.fromList([0.05], [1], DType.float64),
          NDArray.fromList([10.0], [1], DType.float64),
          NDArray.fromList([-1000.0], [1], DType.float64),
          out: outContig,
        );

        // All-1D contiguous (size 2, with r == 0.0 and r != 0.0) for fv, pv, pmt
        final r1d = NDArray.fromList([0.0, 0.05], [2], DType.float64);
        final n1d = NDArray.fromList([10.0, 10.0], [2], DType.float64);
        final p1d = NDArray.fromList([-100.0, -100.0], [2], DType.float64);
        final v1d = NDArray.fromList([-1000.0, -1000.0], [2], DType.float64);
        final f1d = NDArray.fromList([2000.0, 2000.0], [2], DType.float64);

        expect(fv(r1d, n1d, p1d, v1d, when: PaymentDue.begin).size, 2);
        expect(pv(r1d, n1d, p1d, f1d, when: PaymentDue.begin).size, 2);
        expect(pmt(r1d, n1d, v1d, fv: f1d, when: PaymentDue.begin).size, 2);

        // Mixed scalar + 1D contiguous (hits else loop in _computeTVMDirect)
        expect(fv(r1d, n, p, v).size, 2);
        expect(pv(r1d, n, p, f).size, 2);
        expect(pmt(r1d, n, v, fv: f).size, 2);
        expect(pmt(r1d, n, v).size, 2);

        // General strided / broadcasted 2D path (_computeTVMStrided)
        final r2d = NDArray.fromList(
          [0.0, 0.05, 0.0, 0.08],
          [2, 2],
          DType.float64,
        ).transpose(); // non-contiguous!
        final n2d = NDArray.fromList(
          [10.0, 10.0, 20.0, 20.0],
          [2, 2],
          DType.float64,
        );
        final p2d = NDArray.fromList(
          [-100.0, -100.0, -200.0, -200.0],
          [2, 2],
          DType.float64,
        );
        final v2d = NDArray.fromList(
          [-1000.0, -1000.0, -2000.0, -2000.0],
          [2, 2],
          DType.float64,
        );
        final f2d = NDArray.fromList(
          [1000.0, 1000.0, 2000.0, 2000.0],
          [2, 2],
          DType.float64,
        );

        expect(fv(r2d, n2d, p2d, v2d).shape, [2, 2]);
        expect(pv(r2d, n2d, p2d, f2d).shape, [2, 2]);
        expect(pmt(r2d, n2d, v2d, fv: f2d).shape, [2, 2]);
        expect(pmt(r2d, n2d, v2d).shape, [2, 2]);
      });
    });

    test('npv and irr edge cases and multi-root selection', () {
      NDArray.scope(() {
        final disposed = NDArray.scalar(0.05, dtype: DType.float64)..dispose();
        final vals = NDArray.fromList([-100.0, 60.0, 60.0], [3], DType.float64);
        final rScalar = NDArray.scalar(0.05, dtype: DType.float64);

        expect(() => npv(disposed, vals), throwsStateError);
        expect(() => npv(rScalar, disposed), throwsStateError);
        expect(() => npv(rScalar, vals, out: disposed), throwsStateError);
        expect(() => npv(rScalar, rScalar), throwsArgumentError);
        expect(
          () => npv(rScalar, vals, out: NDArray.zeros([2], DType.float64)),
          throwsArgumentError,
        );

        // Scalar rate: out aliasing rate, out aliasing values, contiguous out, N == 0, 1 + r <= 0
        final aliasRate = NDArray.scalar(0.05, dtype: DType.float64);
        npv(aliasRate, vals, out: aliasRate);
        final outScalar = NDArray.scalar(0.0, dtype: DType.float64);
        npv(rScalar, vals, out: outScalar);

        final emptyCashflows = NDArray.create([2, 0], DType.float64);
        expect(npv(rScalar, emptyCashflows).toList(), [0.0, 0.0]);

        final negRate = NDArray.scalar(-1.5, dtype: DType.float64);
        final stridedVals = NDArray.fromList(
          [-100.0, 60.0, 50.0, 60.0],
          [2, 2],
          DType.float64,
        ).transpose();
        expect(npv(negRate, stridedVals).shape, [2]);

        // Multi-element rate with out
        final multiRate = NDArray.fromList([0.05, 0.10], [2], DType.float64);
        final outMulti = NDArray.zeros([2], DType.float64);
        expect(npv(multiRate, vals, out: outMulti).size, 2);

        // irr
        expect(() => irr(disposed), throwsStateError);
        expect(() => irr(vals, out: disposed), throwsStateError);
        expect(
          () => irr(NDArray.zeros([2, 2], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => irr(vals, out: NDArray.zeros([1], DType.float64)),
          throwsArgumentError,
        );

        // All zeros, 1-element non-zero, all negative, NaN first element
        final zeros = NDArray.fromList([0.0, 0.0], [2], DType.float64);
        expect(
          () => irr(zeros, raiseExceptions: true),
          throwsA(isA<NoRealSolutionException>()),
        );
        final irrOut = NDArray.scalar(0.0, dtype: DType.float64);
        expect(irr(zeros, out: irrOut).scalar.isNaN, isTrue);
        expect(irr(zeros).scalar.isNaN, isTrue);

        final singleNonZero = NDArray.fromList([10.0], [1], DType.float64);
        expect(irr(singleNonZero).scalar.isNaN, isTrue);

        final allNeg = NDArray.fromList([-10.0, -20.0], [2], DType.float64);
        expect(irr(allNeg).scalar.isNaN, isTrue);

        final nanVals = NDArray.fromList(
          [double.nan, 10.0],
          [2],
          DType.float64,
        );
        expect(() => irr(nanVals), throwsA(isA<LinAlgException>()));

        // Leading/trailing zeros & out
        final padded = NDArray.fromList(
          [0.0, -100.0, 110.0, 0.0],
          [4],
          DType.float64,
        );
        expect(irr(padded, out: irrOut).scalar, closeTo(0.1, 1e-6));

        // Sign change but only complex roots -> eirr.isEmpty
        final complexOnly = NDArray.fromList(
          [1.0, -1.0, 1.0],
          [3],
          DType.float64,
        );
        expect(
          () => irr(complexOnly, raiseExceptions: true),
          throwsA(isA<NoRealSolutionException>()),
        );
        expect(irr(complexOnly, raiseExceptions: false).scalar.isNaN, isTrue);

        // Root with r < -1.0 (x = -0.5 -> r = -1.5 skipped) and one valid root (x = 1.5 -> r = 0.5)
        final oneBelowMinusOne = NDArray.fromList(
          [1.0, -1.0, -0.75],
          [3],
          DType.float64,
        );
        expect(irr(oneBelowMinusOne).scalar, closeTo(0.5, 1e-6));

        // Multiple real roots:
        // 1. All positive roots: (x - 1.5)(x - 1.2) = x^2 - 2.7x + 1.8 -> r = 0.5, 0.2
        final twoPosRoots = NDArray.fromList(
          [1.0, -2.7, 1.8],
          [3],
          DType.float64,
        );
        expect(irr(twoPosRoots).scalar, closeTo(0.2, 1e-6));

        // 2. All negative roots >= -1: (x - 0.5)(x - 0.8) = x^2 - 1.3x + 0.4 -> r = -0.5, -0.2
        final twoNegRoots = NDArray.fromList(
          [1.0, -1.3, 0.4],
          [3],
          DType.float64,
        );
        expect(irr(twoNegRoots).scalar, closeTo(-0.2, 1e-6));

        // 3. Mixed sign roots: (x - 1.5)(x - 1.2)(x - 0.5) = x^3 - 3.2x^2 + 3.15x - 0.9 -> r = 0.5, 0.2, -0.5
        final mixedRoots = NDArray.fromList(
          [1.0, -3.2, 3.15, -0.9],
          [4],
          DType.float64,
        );
        expect(irr(mixedRoots).scalar, closeTo(0.2, 1e-6));

        // 4. fv/pv overflow and ultra-small rate parity across SIMD & scalar lanes
        final rOverflow = NDArray.fromList(List<double>.filled(8, 0.1), [
          8,
        ], DType.float64);
        final nOverflow = NDArray.fromList(List<double>.filled(8, 20000.0), [
          8,
        ], DType.float64);
        final pOverflow = NDArray.fromList(List<double>.filled(8, -100.0), [
          8,
        ], DType.float64);
        final vOverflow = NDArray.fromList(List<double>.filled(8, -1000.0), [
          8,
        ], DType.float64);
        final fvOv = fv(rOverflow, nOverflow, pOverflow, vOverflow);
        expect(fvOv.getCell([0]), double.infinity);
        expect(fvOv.getCell([7]), double.infinity);
      });
    });
  });
}
