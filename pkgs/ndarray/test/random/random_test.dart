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
  group('Random Module Extra Coverage Tests', () {
    test('uniform exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid dtype
        expect(() => uniform([2, 2], dtype: DType.int32), throwsArgumentError);

        // Incompatible out shape
        expect(
          () =>
              uniform([2, 2], out: NDArray<Float64>.create([3], DType.float64)),
          throwsArgumentError,
        );
        // Incompatible out dtype
        expect(
          () => uniform(
            [2, 2],
            dtype: DType.float64,
            out: NDArray<Float32>.create([2, 2], DType.float32) as dynamic,
          ),
          throwsArgumentError,
        );
      });
    });

    test('randint exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid low >= high
        expect(() => randint([2, 2], low: 10, high: 10), throwsArgumentError);
        // Invalid dtype
        expect(
          () => randint([2, 2], low: 0, high: 5, dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible out shape
        expect(
          () => randint(
            [2, 2],
            low: 0,
            high: 5,
            out: NDArray<Int64>.create([3], DType.int64),
          ),
          throwsArgumentError,
        );
      });
    });

    test('normal exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid scale <= 0.0
        expect(() => normal([2, 2], scale: 0.0), throwsArgumentError);
        // Invalid dtype
        expect(() => normal([2, 2], dtype: DType.int64), throwsArgumentError);

        // Incompatible out shape
        expect(
          () =>
              normal([2, 2], out: NDArray<Float64>.create([3], DType.float64)),
          throwsArgumentError,
        );
      });
    });

    test('exponential exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid scale <= 0.0
        expect(() => exponential([2, 2], scale: 0.0), throwsArgumentError);
        // Invalid dtype
        expect(
          () => exponential([2, 2], dtype: DType.int64),
          throwsArgumentError,
        );

        // Incompatible out shape
        expect(
          () => exponential([
            2,
            2,
          ], out: NDArray<Float64>.create([3], DType.float64)),
          throwsArgumentError,
        );
      });
    });

    test('poisson exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid lam <= 0.0
        expect(() => poisson([2, 2], lam: 0.0), throwsArgumentError);
        // Invalid dtype
        expect(
          () => poisson([2, 2], dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible out shape
        expect(
          () => poisson([2, 2], out: NDArray<Int64>.create([3], DType.int64)),
          throwsArgumentError,
        );
      });
    });

    test('binomial exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid n < 0
        expect(() => binomial([2, 2], n: -1, p: 0.5), throwsArgumentError);
        // Invalid p out of [0, 1]
        expect(() => binomial([2, 2], n: 10, p: 1.5), throwsArgumentError);
        // Invalid dtype
        expect(
          () => binomial([2, 2], n: 10, p: 0.5, dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible out shape
        expect(
          () => binomial(
            [2, 2],
            n: 10,
            p: 0.5,
            out: NDArray<Int64>.create([3], DType.int64),
          ),
          throwsArgumentError,
        );
      });
    });

    test('multivariateNormal exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        final mean = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final cov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        // Invalid cov length/dimensions (non-2D or non-square)
        expect(
          () => multivariateNormal(mean, NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );
        // Invalid dtype
        expect(
          () => multivariateNormal(mean, cov, dtype: DType.int64),
          throwsArgumentError,
        );

        // Incompatible out shape
        expect(
          () => multivariateNormal(
            mean,
            cov,
            size: [2],
            out: NDArray<Float64>.create([3], DType.float64),
          ),
          throwsArgumentError,
        );

        // Valid out recycler buffer
        final validInto = NDArray<Float64>.create([2, 2], DType.float64);
        final res = multivariateNormal(mean, cov, size: [2], out: validInto);
        expect(res == validInto, true);
        expect(res.shape, [2, 2]);
      });
    });

    test('multinomial exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        final pvals = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);

        // Invalid dtype
        expect(
          () => multinomial(10, pvals, dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible out shape
        expect(
          () => multinomial(
            10,
            pvals,
            size: [2],
            out: NDArray<Int32>.create([3], DType.int32),
          ),
          throwsArgumentError,
        );

        // Valid out recycler buffer
        final validInto = NDArray<Int32>.create([2, 3], DType.int32);
        final res = multinomial(10, pvals, size: [2], out: validInto);
        expect(res == validInto, true);
        expect(res.shape, [2, 3]);
      });
    });

    test('FFT float32 and exponential FFI acceleration correctness', () {
      NDArray.scope(() {
        // 1. uniform float32 (FFI)
        final u32 = uniform([100], dtype: DType.float32);
        expect(u32.dtype, DType.float32);
        expect(u32.shape, [100]);
        final u32List = u32.toList();
        for (var i = 0; i < 100; i++) {
          expect(u32List[i], greaterThanOrEqualTo(0.0));
          expect(u32List[i], lessThan(1.0));
        }

        // 2. exponential float64
        final exp64 = exponential([50], scale: 2.0, dtype: DType.float64);
        expect(exp64.dtype, DType.float64);
        expect(exp64.shape, [50]);
        final exp64List = exp64.toList();
        for (var i = 0; i < 50; i++) {
          expect(exp64List[i], greaterThanOrEqualTo(0.0));
        }

        // 3. exponential float32
        final exp32 = exponential([50], scale: 1.5, dtype: DType.float32);
        expect(exp32.dtype, DType.float32);
        expect(exp32.shape, [50]);
        final exp32List = exp32.toList();
        for (var i = 0; i < 50; i++) {
          expect(exp32List[i], greaterThanOrEqualTo(0.0));
        }
      });
    });

    test('secure random generation across all 8 distributions', () {
      NDArray.scope(() {
        // 1. uniform
        final uniSec = uniform([10], dtype: DType.float64, secure: true);
        expect(uniSec.shape, [10]);
        final uniSecList = uniSec.toList();
        for (var i = 0; i < 10; i++) {
          expect(uniSecList[i], greaterThanOrEqualTo(0.0));
          expect(uniSecList[i], lessThan(1.0));
        }

        // 2. randint
        final riSec = randint(
          [10],
          low: -5,
          high: 15,
          dtype: DType.int32,
          secure: true,
        );
        expect(riSec.shape, [10]);
        final riSecList = riSec.toList();
        for (var i = 0; i < 10; i++) {
          expect(riSecList[i], greaterThanOrEqualTo(-5));
          expect(riSecList[i], lessThan(15));
        }

        // 2b. randint with huge range (> 2^32)
        final riSecHuge = randint(
          [5],
          low: 0,
          high: 100000000000000,
          dtype: DType.int64,
          secure: true,
        );
        expect(riSecHuge.shape, [5]);

        // 3. normal
        final normSec = normal(
          [10],
          loc: 5.0,
          scale: 2.0,
          dtype: DType.float32,
          secure: true,
        );
        expect(normSec.shape, [10]);
        expect(normSec.dtype, DType.float32);

        // 4. exponential
        final expSec = exponential(
          [10],
          scale: 3.0,
          dtype: DType.float64,
          secure: true,
        );
        expect(expSec.shape, [10]);
        final expSecList = expSec.toList();
        for (var i = 0; i < 10; i++) {
          expect(expSecList[i], greaterThanOrEqualTo(0.0));
        }

        // 5. poisson
        final poiSec = poisson(
          [10],
          lam: 4.0,
          dtype: DType.int64,
          secure: true,
        );
        expect(poiSec.shape, [10]);
        final poiSecList = poiSec.toList();
        for (var i = 0; i < 10; i++) {
          expect(poiSecList[i], greaterThanOrEqualTo(0));
        }

        // 6. binomial (small n)
        final binSecSmall = binomial(
          [10],
          n: 10,
          p: 0.5,
          dtype: DType.int32,
          secure: true,
        );
        expect(binSecSmall.shape, [10]);
        final binSecSmallList = binSecSmall.toList();
        for (var i = 0; i < 10; i++) {
          expect(binSecSmallList[i], greaterThanOrEqualTo(0));
          expect(binSecSmallList[i], lessThanOrEqualTo(10));
        }

        // 7. binomial (large n)
        final binSecLarge = binomial(
          [10],
          n: 100,
          p: 0.4,
          dtype: DType.int64,
          secure: true,
        );
        expect(binSecLarge.shape, [10]);
        final binSecLargeList = binSecLarge.toList();
        for (var i = 0; i < 10; i++) {
          expect(binSecLargeList[i], greaterThanOrEqualTo(0));
          expect(binSecLargeList[i], lessThanOrEqualTo(100));
        }

        // 7b. binomial (large n) with int32
        final binSecLarge32 = binomial(
          [5],
          n: 100,
          p: 0.35,
          dtype: DType.int32,
          secure: true,
        );
        expect(binSecLarge32.shape, [5]);
        expect(binSecLarge32.dtype, DType.int32);

        // 8. multivariateNormal
        final mean = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final cov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final mvSec = multivariateNormal(mean, cov, size: [5], secure: true);
        expect(mvSec.shape, [5, 2]);

        // 9. multinomial
        final pvals = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);
        final multiSec = multinomial(10, pvals, size: [5], secure: true);
        expect(multiSec.shape, [5, 3]);
      });
    });

    group('Random Sampling & Types Expansion Tests', () {
      test('randint support for uint8 and int16', () {
        NDArray.scope(() {
          // uint8 randint
          final u8 = randint([10], low: 5, high: 150, dtype: DType.uint8);
          expect(u8.dtype, DType.uint8);
          expect(u8.shape, [10]);
          final u8List = u8.toList();
          for (var i = 0; i < 10; i++) {
            expect(u8List[i], greaterThanOrEqualTo(5));
            expect(u8List[i], lessThan(150));
          }

          // int16 randint
          final i16 = randint([10], low: -1000, high: 1000, dtype: DType.int16);
          expect(i16.dtype, DType.int16);
          expect(i16.shape, [10]);
          final i16List = i16.toList();
          for (var i = 0; i < 10; i++) {
            expect(i16List[i], greaterThanOrEqualTo(-1000));
            expect(i16List[i], lessThan(1000));
          }

          // secure randint uint8
          final u8Sec = randint(
            [5],
            low: 0,
            high: 255,
            dtype: DType.uint8,
            secure: true,
          );
          expect(u8Sec.dtype, DType.uint8);

          // secure randint int16
          final i16Sec = randint(
            [5],
            low: -32768,
            high: 32767,
            dtype: DType.int16,
            secure: true,
          );
          expect(i16Sec.dtype, DType.int16);
        });
      });

      test('shuffle 1D and N-D arrays', () {
        NDArray.scope(() {
          // 1D shuffle
          final a1 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );
          shuffle(a1, seed: 42);
          expect(a1.shape, [5]);
          expect(a1.toList(), containsAll([1.0, 2.0, 3.0, 4.0, 5.0]));

          // ND shuffle along axis 0
          final a2 = NDArray.fromList(
            [
              1.0, 1.0, // row 0
              2.0, 2.0, // row 1
              3.0, 3.0, // row 2
            ],
            [3, 2],
            DType.float64,
          );
          shuffle(a2, seed: 100);
          expect(a2.shape, [3, 2]);
          final list = a2.toList();
          // Check that rows are preserved as units
          expect(list[0] == list[1], true);
          expect(list[2] == list[3], true);
          expect(list[4] == list[5], true);
          expect(list, containsAll([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));
        });
      });

      test('permutation along axis 0', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final perm = permutation(a, seed: 42);
          expect(perm != a, true); // Brand new copy
          expect(perm.shape, [4]);
          expect(perm.toList(), containsAll([1, 2, 3, 4]));
          expect(a.toList(), [1, 2, 3, 4]); // Original unchanged
        });
      });

      test('choice sampling with and without replacement', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5],
            DType.float64,
          );

          // Choice with replacement
          final c1 = choice(a, size: [10], replace: true, seed: 42);
          expect(c1.shape, [10]);
          final c1List = c1.toList();
          final aList = a.toList();
          for (var i = 0; i < 10; i++) {
            expect(aList, contains(c1List[i]));
          }

          // Choice without replacement
          final c2 = choice(a, size: [3], replace: false, seed: 42);
          expect(c2.shape, [3]);
          expect(c2.toList().toSet().length, 3); // All unique

          // Choice with probabilities
          final p = NDArray.fromList(
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [5],
            DType.float64,
          );
          final c3 = choice(a, size: [5], replace: true, p: p, seed: 42);
          expect(c3.toList(), [
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
          ]); // 100% probability on 20.0

          // Choice without replacement and non-uniform probabilities
          final c4 = choice(a, size: [2], replace: false, p: p, seed: 42);
          expect(c4.shape, [2]);
          expect(c4.toList(), contains(20.0)); // Must contain 20.0
        });
      });

      test('choice and shuffle edge cases and exceptions', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final disposed = NDArray<int>.create([3], DType.int32)..dispose();

          // Disposed exceptions
          expect(() => choice(disposed), throwsStateError);
          expect(() => shuffle(disposed), throwsStateError);
          expect(() => permutation(disposed), throwsStateError);

          // Non 1-D input for choice
          final a2D = NDArray.zeros([2, 2], DType.float64);
          expect(() => choice(a2D), throwsArgumentError);

          // Mismatched probability size
          final pBad = NDArray.fromList([0.5, 0.5], [2], DType.float64);
          expect(() => choice(a, p: pBad), throwsArgumentError);

          // Probability contains negative values
          final pNeg = NDArray.fromList([-0.2, 0.8, 0.4], [3], DType.float64);
          expect(() => choice(a, p: pNeg), throwsArgumentError);

          // Choice without replacement where size > a.size
          expect(
            () => choice(a, size: [4], replace: false),
            throwsArgumentError,
          );

          // Out buffer mismatch
          final outBad = NDArray<int>.create([4], DType.int32);
          expect(() => choice(a, size: [3], out: outBad), throwsArgumentError);
          final outDisposed = NDArray<int>.create([3], DType.int32)..dispose();
          expect(
            () => choice(a, size: [3], out: outDisposed),
            throwsStateError,
          );
          expect(() => permutation(a, out: outDisposed), throwsStateError);
        });
      });

      test('shuffle, permutation, and choice with various data types', () {
        NDArray.scope(() {
          // Complex128
          final c128 = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(3.0, 4.0),
              Complex(5.0, 6.0),
              Complex(7.0, 8.0),
            ],
            [4],
            DType.complex128,
          );
          final c128Perm = permutation(c128, seed: 123);
          expect(c128Perm.dtype, DType.complex128);
          expect(c128Perm.toList(), containsAll(c128.toList()));
          final c128Sample = choice(c128, size: [2], replace: false, seed: 123);
          expect(c128Sample.shape, [2]);
          expect(c128Sample.dtype, DType.complex128);

          // Complex64
          final c64 = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0), Complex(5.0, 6.0)],
            [3],
            DType.complex64,
          );
          shuffle(c64, seed: 99);
          expect(
            c64.toList(),
            containsAll([
              Complex(1.0, 2.0),
              Complex(3.0, 4.0),
              Complex(5.0, 6.0),
            ]),
          );

          // Uint8 / Bool / Int16 / Int64 / Float32
          final u8 = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.uint8);
          shuffle(u8, seed: 42);
          expect(u8.toList(), containsAll([10, 20, 30, 40, 50]));

          final i64 = NDArray.fromList([100, 200, 300, 400], [4], DType.int64);
          final i64Sample = choice(i64, size: [6], replace: true, seed: 42);
          expect(i64Sample.shape, [6]);
          expect(i64Sample.dtype, DType.int64);

          final f32 = NDArray.fromList([1.5, 2.5, 3.5], [3], DType.float32);
          final f32Out = NDArray<double>.create([3], DType.float32);
          permutation(f32, seed: 42, out: f32Out);
          expect(f32Out.toList(), containsAll([1.5, 2.5, 3.5]));
        });
      });

      test('shuffle and choice on strided arrays and views', () {
        NDArray.scope(() {
          // Strided 1D view from a 2D matrix column
          final matrix = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [4, 2],
            DType.float64,
          );

          // Slices along axis 0
          shuffle(matrix, seed: 42);
          expect(matrix.shape, [4, 2]);
          final rowPairs = [
            for (var r = 0; r < 4; r++)
              [
                matrix.getCell([r, 0]),
                matrix.getCell([r, 1]),
              ],
          ];
          expect(
            rowPairs,
            containsAll([
              [1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0],
              [7.0, 8.0],
            ]),
          );

          // 3D array shuffling
          final t3d = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [3, 2, 2],
            DType.float64,
          );
          shuffle(t3d, seed: 100);
          expect(t3d.shape, [3, 2, 2]);

          // Choice with out recycler buffer
          final pool = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);
          final outChoice = NDArray<int>.create([2, 3], DType.int32);
          choice(pool, size: [2, 3], replace: true, seed: 42, out: outChoice);
          expect(outChoice.shape, [2, 3]);
          for (var i = 0; i < 6; i++) {
            expect(pool.toList(), contains(outChoice.getCellFlat(i)));
          }
        });
      });

      test('shuffle 0-length and 1-length arrays', () {
        NDArray.scope(() {
          final a0 = NDArray<double>.create([0], DType.float64);
          shuffle(a0);
          expect(a0.size, 0);

          final a1 = NDArray.fromList([42.0], [1], DType.float64);
          shuffle(a1);
          expect(a1.toList(), [42.0]);
        });
      });
    });
  });

  group('NDArray Random Distributions Tests', () {
    group('normal distribution tests', () {
      test(
        'Shape, type, and standard parameter checks',
        () => NDArray.scope(() {
          final a = normal([2, 5], loc: 0.0, scale: 1.0, dtype: DType.float64);
          expect(a.shape, [2, 5]);
          expect(a.dtype, DType.float64);
          expect(a.size, 10);
        }),
      );

      test(
        'Statistical mean and variance validation (Law of Large Numbers)',
        () => NDArray.scope(() {
          const loc = 10.0;
          const scale = 3.0;
          // Draw a large sample size to confirm sample parameters match target distributions
          final a = normal(
            [10000],
            loc: loc,
            scale: scale,
            dtype: DType.float64,
            seed: 42,
          );

          final sampleMean = mean(a).scalar;
          final sampleStd = std(a).scalar;

          // Over 10,000 draws, sample mean should be very close to loc, and stddev close to scale!
          expect(sampleMean, closeTo(loc, 0.1));
          expect(sampleStd, closeTo(scale, 0.1));
        }),
      );

      test(
        'Throws on invalid parameters',
        () => NDArray.scope(() {
          expect(() => normal([2], scale: 0.0), throwsArgumentError);
          expect(() => normal([2], scale: -1.5), throwsArgumentError);
          expect(() => normal([2], dtype: DType.int32), throwsArgumentError);
        }),
      );
    });

    group('exponential distribution tests', () {
      test(
        'Basic property and statistics validation',
        () => NDArray.scope(() {
          const scale = 2.5;
          final a = exponential(
            [5000],
            scale: scale,
            dtype: DType.float64,
            seed: 42,
          );

          expect(a.shape, [5000]);
          // All values in an exponential distribution are strictly non-negative
          for (final val in a.toList()) {
            expect(val, greaterThanOrEqualTo(0.0));
          }

          final sampleMean = mean(a).scalar;
          // Exponential distribution mean is exactly equal to its scale parameter (beta)
          expect(sampleMean, closeTo(scale, 0.15));
        }),
      );
    });

    group('poisson distribution tests', () {
      test(
        'Small lambda exact Knuth path checks',
        () => NDArray.scope(() {
          const lam = 4.0;
          final a = poisson([5000], lam: lam, dtype: DType.int64, seed: 42);
          expect(a.dtype, DType.int64);

          final sampleMean = mean(a).scalar;
          final sampleVar = variance(a).scalar;

          // In a Poisson distribution, both Mean and Variance are exactly equal to lambda!
          expect(sampleMean, closeTo(lam, 0.15));
          expect(sampleVar, closeTo(lam, 0.25));
        }),
      );

      test(
        'Large lambda Gaussian Approximation path triggers safely',
        () => NDArray.scope(() {
          const lam = 50.0;
          // Large lam triggers Track B to avoid numerical underflow or infinite loops!
          final a = poisson([5000], lam: lam, dtype: DType.int32, seed: 42);
          expect(a.dtype, DType.int32);

          final sampleMean = mean(a).scalar;
          expect(sampleMean, closeTo(lam, 0.3));
          for (final val in a.toList()) {
            expect(val, greaterThanOrEqualTo(0));
          }
        }),
      );
    });

    group('binomial distribution tests', () {
      test(
        'Small n Bernoulli trials path checks',
        () => NDArray.scope(() {
          const n = 20;
          const p = 0.4;
          final a = binomial([5000], n: n, p: p, dtype: DType.int64, seed: 42);

          expect(a.dtype, DType.int64);
          for (final val in a.toList()) {
            expect(val, greaterThanOrEqualTo(0));
            expect(val, lessThanOrEqualTo(n));
          }

          final sampleMean = mean(a).scalar;
          final expectedMean = n * p; // 20 * 0.4 = 8.0
          expect(sampleMean, closeTo(expectedMean, 0.2));
        }),
      );

      test(
        'Large n Normal Approximation triggers safely',
        () => NDArray.scope(() {
          const n = 1000;
          const p = 0.3;
          final a = binomial([5000], n: n, p: p, dtype: DType.int32, seed: 42);

          expect(a.dtype, DType.int32);
          final sampleMean = mean(a).scalar;
          final expectedMean = n * p; // 1000 * 0.3 = 300.0
          expect(sampleMean, closeTo(expectedMean, 1.0));
        }),
      );
    });

    group('Seeded Reproducibility and Determinism checks', () {
      test(
        'Seeding Random(seed) guarantees exact reproducible draws',
        () => NDArray.scope(() {
          final shape = [2, 3];

          // Use exact FFI seeds for C-land reproducibility
          final n1 = normal(shape, loc: 5.0, scale: 2.0, seed: 12345);
          final n2 = normal(shape, loc: 5.0, scale: 2.0, seed: 12345);
          expect(n1.toList(), n2.toList()); // exact matching doubles!

          final e1 = exponential([10], scale: 1.5, seed: 42);
          final e2 = exponential([10], scale: 1.5, seed: 42);
          expect(e1.toList(), e2.toList());

          final p1 = poisson([10], lam: 12.0, seed: 7);
          final p2 = poisson([10], lam: 12.0, seed: 7);
          expect(p1.toList(), p2.toList());

          final b1 = binomial([10], n: 100, p: 0.2, seed: 999);
          final b2 = binomial([10], n: 100, p: 0.2, seed: 999);
          expect(b1.toList(), b2.toList());
        }),
      );
    });
    test(
      'randint() supporting wide ranges (> 2^32)',
      () => NDArray.scope(() {
        // Generate a 64-bit integer range that exceeds 2^32
        final a = randint(
          [10],
          low: 100000000000,
          high: 200000000000,
          dtype: DType.int64,
        );
        expect(a.shape, [10]);
        expect(a.dtype, DType.int64);
        final aList = a.toList();
        for (final val in aList) {
          expect(val, greaterThanOrEqualTo(100000000000));
          expect(val, lessThan(200000000000));
        }
      }),
    );

    test(
      'exponential() supporting lam (rate) parameter alias',
      () => NDArray.scope(() {
        final a = exponential([10], lam: 2.0); // lam = 2.0 -> scale = 0.5
        expect(a.shape, [10]);
        expect(a.dtype, DType.float64);
        final aList = a.toList();
        for (final val in aList) {
          expect(val, greaterThanOrEqualTo(0.0));
        }

        expect(() => exponential([10], lam: -1.0), throwsArgumentError);
      }),
    );

    test(
      'uniform() validation checks coverage',
      () => NDArray.scope(() {
        expect(
          () => uniform([5], dtype: DType.int64 as dynamic),
          throwsArgumentError,
        );
      }),
    );

    test(
      'randint() validation checks coverage',
      () => NDArray.scope(() {
        expect(
          () => randint([5], low: 0, high: 10, dtype: DType.float64 as dynamic),
          throwsArgumentError,
        );
        expect(() => randint([5], low: 10, high: 5), throwsArgumentError);
      }),
    );

    test(
      'normal() validation and zero-avoidance checks coverage',
      () => NDArray.scope(() {
        expect(() => normal([5], scale: 0.0), throwsArgumentError);

        final a = normal([2]);
        expect(a.shape, [2]);
        expect(a.dtype, DType.float64);
      }),
    );

    test(
      'exponential() validation checks coverage',
      () => NDArray.scope(() {
        expect(
          () => exponential([5], dtype: DType.int32 as dynamic),
          throwsArgumentError,
        );
      }),
    );

    test(
      'poisson() validation, large lambda, and zero-avoidance checks',
      () => NDArray.scope(() {
        expect(
          () => poisson([5], dtype: DType.float64 as dynamic),
          throwsArgumentError,
        );
        expect(() => poisson([5], lam: 0.0), throwsArgumentError);

        final a = poisson([2], lam: 40.0);
        expect(a.shape, [2]);
        expect(a.dtype, DType.int64);
      }),
    );

    test('binomial() validation, zero trials, zero stddev, and zero-avoidance checks', () {
      expect(
        () => binomial([5], n: 10, p: 0.5, dtype: DType.float64 as dynamic),
        throwsArgumentError,
      );
      expect(() => binomial([5], n: -1, p: 0.5), throwsArgumentError);
      expect(() => binomial([5], n: 10, p: 1.5), throwsArgumentError);

      final zeroTrials = binomial([5], n: 0, p: 0.5);
      expect(zeroTrials.toList(), [0, 0, 0, 0, 0]);

      final zeroStddev = binomial([5], n: 100, p: 1.0);
      expect(zeroStddev.toList(), [100, 100, 100, 100, 100]);

      final a = binomial([2], n: 100, p: 0.5);
      expect(a.shape, [2]);
      expect(a.dtype, DType.int64);

      final oddLength = binomial([5], n: 100, p: 0.5);
      expect(oddLength.shape, [5]);
      expect(oddLength.dtype, DType.int64);
    });

    test(
      'uniform() and normal() Float32, and randint() Int32 coverage',
      () => NDArray.scope(() {
        // 1. uniform with float32
        final u = uniform([10], dtype: DType.float32);
        expect(u.shape, [10]);
        expect(u.dtype, DType.float32);
        final uList = u.toList();
        for (final val in uList) {
          expect(val, greaterThanOrEqualTo(0.0));
          expect(val, lessThan(1.0));
        }

        // 2. randint with int32
        final ri = randint([10], low: 0, high: 5, dtype: DType.int32);
        expect(ri.shape, [10]);
        expect(ri.dtype, DType.int32);
        final riList = ri.toList();
        for (final val in riList) {
          expect(val, greaterThanOrEqualTo(0));
          expect(val, lessThan(5));
        }

        // 3. normal with float32
        final n = normal([10], dtype: DType.float32);
        expect(n.shape, [10]);
        expect(n.dtype, DType.float32);
      }),
    );

    test(
      'Multivariate Normal Distribution multivariateNormal() correctness',
      () => NDArray.scope(() {
        final mean = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        // Identity covariance Sigma = I
        final cov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        // Draw 500 samples (returns shape [500, 2])
        final samples = multivariateNormal(mean, cov, size: [500]);

        expect(samples.shape, [500, 2]);
        expect(samples.dtype, DType.float64);

        // Check statistical means of the drawn samples to verify standard convergence!
        var sumX = 0.0;
        var sumY = 0.0;
        for (var i = 0; i < 500; i++) {
          sumX += samples.getCell([i, 0]);
          sumY += samples.getCell([i, 1]);
        }
        final meanX = sumX / 500;
        final meanY = sumY / 500;

        // Means should converge closely to [10, 20] under standard deviation 1
        expect(meanX, closeTo(10.0, 0.5));
        expect(meanY, closeTo(20.0, 0.5));

        // Verify ArgumentError throwing exceptions
        final badMean = NDArray.fromList([10.0, 20.0], [1, 2], DType.float64);
        expect(() => multivariateNormal(badMean, cov), throwsArgumentError);

        final mismatchedMean = NDArray.fromList([10.0], [1], DType.float64);
        expect(
          () => multivariateNormal(mismatchedMean, cov),
          throwsArgumentError,
        );
      }),
    );

    test(
      'Multinomial Distribution multinomial() trial simulation correctness',
      () {
        final pvals = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);

        // Draw 1000 samples of 10 trials (shape [1000, 3])
        final samples = multinomial(10, pvals, size: [1000]);

        expect(samples.shape, [1000, 3]);
        expect(samples.dtype, DType.int32);

        // Test multinomial with pvals requiring normalization (does not sum to 1.0)
        final nonNormalizedPvals = NDArray.fromList(
          [0.2, 0.6, 0.3],
          [3],
          DType.float64,
        );
        final samplesNonNorm = multinomial(10, nonNormalizedPvals, size: [5]);
        expect(samplesNonNorm.shape, [5, 3]);

        // For every sample, the sum of category counts must exactly equal the trials 'n' (10)!
        for (var i = 0; i < 1000; i++) {
          final sum =
              samples.getCell([i, 0]) +
              samples.getCell([i, 1]) +
              samples.getCell([i, 2]);
          expect(sum, 10);
        }

        // Statistical ratios should converge close to [0.2, 0.5, 0.3]
        var count0 = 0.0;
        var count1 = 0.0;
        var count2 = 0.0;
        for (var i = 0; i < 1000; i++) {
          count0 += samples.getCell([i, 0]);
          count1 += samples.getCell([i, 1]);
          count2 += samples.getCell([i, 2]);
        }
        final r0 = count0 / 10000.0;
        final r1 = count1 / 10000.0;
        final r2 = count2 / 10000.0;

        expect(r0, closeTo(0.2, 0.05));
        expect(r1, closeTo(0.5, 0.05));
        expect(r2, closeTo(0.3, 0.05));

        // Verify exceptions throwing
        expect(
          () => multinomial(-5, pvals),
          throwsArgumentError,
        ); // negative trials

        final badShapePvals = NDArray.fromList(
          [0.5, 0.5],
          [1, 2],
          DType.float64,
        );
        expect(
          () => multinomial(10, badShapePvals),
          throwsArgumentError,
        ); // bad shape pvals

        final negativePvals = NDArray.fromList([-0.2, 1.2], [2], DType.float64);
        expect(
          () => multinomial(10, negativePvals),
          throwsArgumentError,
        ); // negative probability
      },
    );

    test(
      'Random generators support out parameter recycler buffer reuse',
      () => NDArray.scope(() {
        final buffer = NDArray<Float64>.zeros([10], DType.float64);

        // 1. uniform out
        final u = uniform([10], out: buffer);
        expect(identical(u, buffer), true);

        // 2. normal out
        final n = normal([10], out: buffer);
        expect(identical(n, buffer), true);

        // 3. exponential out
        final e = exponential([10], out: buffer);
        expect(identical(e, buffer), true);

        // 4. randint out
        final bufferInt = NDArray<Int64>.zeros([10], DType.int64);
        final r = randint([10], low: 0, high: 10, out: bufferInt);
        expect(identical(r, bufferInt), true);
      }),
    );
  });
}
