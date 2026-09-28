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
import 'dart:math' as math;
import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  // Limit OpenBLAS execution to 1 thread to avoid parallel thread context switch
  // overhead on lightweight/small matrices in the benchmark suite.
  setNumThreads(1);

  await criterion(
    'ndarray ALL-INCLUSIVE PERFORMANCE BENCHMARK SUITE MASTER',
    (c) {
      // ============================================================================
      // TRACK A: PROBABILITY DISTRIBUTIONS & RNG TRACK
      // ============================================================================
      c.group('Track A: Random Distributions & RNG Solvers', () {
        c.bench('Seeded normal() (Gaussian samples) [size=50,000]', () {
          final arr = normal([50000], seed: 42);
          blackhole(arr);
          arr.dispose();
        }, throughput: Throughput.elements(50000));

        c.bench('Seeded poisson() (Knuth vs Gaussian) [size=20,000]', () {
          final arr = poisson([20000], lam: 35.0, seed: 42);
          blackhole(arr);
          arr.dispose();
        }, throughput: Throughput.elements(20000));

        c.bench('Seeded binomial() (Bernoulli vs Normal) [size=20,000]', () {
          // n >= 50 triggers Normal distribution approximation track natively
          final arr = binomial([20000], n: 60, p: 0.4, seed: 42);
          blackhole(arr);
          arr.dispose();
        }, throughput: Throughput.elements(20000));
      });

      // ============================================================================
      // TRACK B: SORTING & SEARCHING TRACK
      // ============================================================================
      c.group('Track B: Native C Heap Sorting & Searching Broadcasts', () {
        // Shared buffers for Track B
        final templateContig = Float64List(30000);
        for (var i = 0; i < 30000; i++) {
          templateContig[i] = (30000 - i).toDouble();
        }
        final rand = math.Random(42);
        final templateRandom = Float64List.fromList(
          List.generate(30000, (_) => rand.nextDouble()),
        );

        c.bench<NDArray<DTypeTag>>(
          'Native C Heap sort() (Contiguous vector) [size=30,000]',
          (arr) {
            final res = sort(arr);
            blackhole(res);
            res.dispose();
            arr.dispose();
          },
          setup: () => NDArray<DTypeTag>.fromList(templateContig, [
            30000,
          ], DType.float64),
          batchSize: BatchSize.largeInput,
          throughput: Throughput.elements(30000),
        );

        c.bench<NDArray<DTypeTag>>(
          'Native C Heap sort() (Random vector) [size=30,000]',
          (arr) {
            final res = sort(arr);
            blackhole(res);
            res.dispose();
            arr.dispose();
          },
          setup: () => NDArray<DTypeTag>.fromList(templateRandom, [
            30000,
          ], DType.float64),
          batchSize: BatchSize.largeInput,
          throughput: Throughput.elements(30000),
        );

        final maskIndexTarget = NDArray.zeros([100000], DType.float64);
        final maskData = List.generate(100000, (_) => rand.nextBool());
        final mask = NDArray.fromList(maskData, [100000], DType.boolean);

        c.bench('Boolean Mask Advanced Indexing [size=100,000]', () {
          final res = maskIndexTarget[mask];
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(100000));

        c.bench<NDArray<DTypeTag>>(
          'Argsort (argsort) [size=30,000]',
          (arr) {
            final indices = argsort(arr);
            blackhole(indices);
            indices.dispose();
            arr.dispose();
          },
          setup: () => NDArray<DTypeTag>.fromList(templateContig, [
            30000,
          ], DType.float64),
          batchSize: BatchSize.largeInput,
          throughput: Throughput.elements(30000),
        );

        final whereCond = NDArray.zeros([100, 100], DType.boolean);
        final whereX = NDArray.ones([100], DType.float64);
        final whereY = NDArray.ones([100, 100], DType.float64);
        final whereOut = NDArray.zeros([100, 100], DType.float64);

        c.bench('Ternary where() 3-Way Broadcasting [shape=100x100]', () {
          where(whereCond, whereX, whereY, whereOut);
        }, throughput: Throughput.elements(10000));
      });

      // ============================================================================
      // TRACK C: ADVANCED LINEAR ALGEBRA & SIGNALS TRACK
      // ============================================================================
      c.group(
        'Track C: OpenBLAS Linear Algebra & Native pocketfft Signals',
        () {
          final invA = NDArray.eye(100, DType.float64);
          c.bench('OpenBLAS LU Matrix Inversion (inv) [shape=100x100]', () {
            final res = inv(invA);
            blackhole(res);
            res.dispose();
          });

          final qrA = NDArray.zeros([30, 30], DType.float64);
          for (var i = 0; i < 30; i++) {
            for (var j = 0; j < 30; j++) {
              qrA.data[i * 30 + j] = (i + j + 1.0) / 10.0;
              if (i == j) {
                qrA.data[i * 30 + j] = qrA.data[i * 30 + j] + 1.0;
              }
            }
          }
          c.bench('QR Decomposition (qr) [shape=30x30]', () {
            final res = qr(qrA);
            blackhole(res);
            res.q.dispose();
            res.r.dispose();
          });

          c.bench('SVD Decomposition (svd) [shape=30x30]', () {
            final res = svd(qrA);
            blackhole(res);
            res.dispose();
          });

          final signalFft = NDArray.zeros([2048], DType.float64);
          c.bench('Native Mixed-Radix C FFI pocketfft (fft) [length=2048]', () {
            final res = fft(signalFft);
            blackhole(res);
            res.dispose();
          }, throughput: Throughput.elements(2048));

          final cholA = NDArray.zeros([30, 30], DType.float64);
          for (var i = 0; i < 30; i++) {
            for (var j = 0; j < 30; j++) {
              cholA.data[i * 30 + j] = (i + j + 1.0) / 10.0;
              if (i == j) {
                cholA.data[i * 30 + j] = cholA.data[i * 30 + j] + 30.0;
              }
            }
          }
          c.bench('Cholesky Decomposition (cholesky) [shape=30x30]', () {
            final res = cholesky(cholA);
            blackhole(res);
            res.dispose();
          });

          final matA = NDArray.ones([100, 100], DType.float64);
          final matB = NDArray.ones([100, 100], DType.float64);
          c.bench('Matrix Multiplication (matmul) [shape=100x100]', () {
            final res = matmul(matA, matB);
            blackhole(res);
            res.dispose();
          });
        },
      );

      // ============================================================================
      // TRACK D: UNIVERSAL UFUNCS, REDUCTIONS & MEMORY TRACK
      // ============================================================================
      c.group('Track D: Universal Ufuncs, Reductions & Memory Strides', () {
        final vec300kX = NDArray.ones([300000], DType.float64);
        final vec300kY = NDArray.ones([300000], DType.float64);
        final out300k = NDArray.create([300000], DType.float64);

        c.bench('Element-wise Same-Shape add(x, y) [size=300,000]', () {
          add(vec300kX, vec300kY, out: out300k);
        }, throughput: Throughput.elements(300000));

        final scalarArr = NDArray.fromList(Float64List.fromList([5.0]), [
          1,
        ], DType.float64);
        c.bench('Scalar Array Broadcast add(x, scalar) [size=300,000]', () {
          add(vec300kX, scalarArr, out: out300k);
        }, throughput: Throughput.elements(300000));

        final vec100k = NDArray.ones([100000], DType.float64);
        final out100k = NDArray.create([100000], DType.float64);

        c.bench('Universal math function sin(x) [size=100,000]', () {
          sin(vec100k, out: out100k);
        }, throughput: Throughput.elements(100000));

        c.bench('Universal math function cos(x) [size=100,000]', () {
          cos(vec100k, out: out100k);
        }, throughput: Throughput.elements(100000));

        c.bench('Universal math function exp(x) [size=100,000]', () {
          exp(vec100k, out: out100k);
        }, throughput: Throughput.elements(100000));

        c.bench('Universal math function clip(x) [size=300,000]', () {
          clip(vec300kX, min: 0.0, max: 0.5, out: out300k);
        }, throughput: Throughput.elements(300000));

        c.bench('Flat Memory Reduction walk sum(x) [size=300,000]', () {
          final s = sum(vec300kX);
          blackhole(s);
        }, throughput: Throughput.elements(300000));

        c.bench('Zeros Array Creation (zeros) [size=1,000,000]', () {
          final arr = NDArray<DTypeTag>.zeros([1000, 1000], DType.float64);
          blackhole(arr);
          arr.dispose();
        }, throughput: Throughput.elements(1000000));

        final concA = NDArray.ones([500000], DType.float64);
        final concB = NDArray.ones([500000], DType.float64);
        c.bench('Flat Array Concatenation (concatenate) [size=1,000,000]', () {
          final res = concatenate([concA, concB], axis: 0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(1000000));

        final parent600k = NDArray.ones([600000], DType.float64);
        final slice300k = parent600k.slice([Slice(start: 0, stop: 300000)]);

        c.bench('Contiguous View Flatten (flatten) [size=300,000]', () {
          final res = slice300k.flatten();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(300000));

        c.bench('Contiguous View Sum Reduction (sum) [size=300,000]', () {
          final s = sum(slice300k);
          blackhole(s);
        }, throughput: Throughput.elements(300000));

        final parentStridedX = NDArray.ones([500, 500], DType.float64);
        final parentStridedY = NDArray.ones([500, 500], DType.float64);
        final xView = parentStridedX.transposed;
        final yView = parentStridedY.transposed;
        final outStrided = NDArray.create([500, 500], DType.float64);

        c.bench('Strided non-contiguous add(x, y) [shape=500x500]', () {
          add(xView, yView, out: outStrided);
        }, throughput: Throughput.elements(250000));
      });

      // ============================================================================
      // TRACK E: DISTANCE METRICS TRACK (pdist & cdist)
      // ============================================================================
      c.group('Track E: Distance Metrics (pdist & cdist)', () {
        final distMatA = normal<Float64>(
          [500, 100],
          dtype: DType.float64,
          seed: 42,
        );
        final distMatB = normal<Float64>(
          [500, 100],
          dtype: DType.float64,
          seed: 43,
        );
        final distIntA = randint(
          [500, 100],
          low: 0,
          high: 2,
          dtype: DType.int32,
          seed: 42,
        );
        final distIntB = randint(
          [500, 100],
          low: 0,
          high: 2,
          dtype: DType.int32,
          seed: 43,
        );

        c.bench('pdist Euclidean [shape=500x100]', () {
          final res = pdist(distMatA, metric: DistanceMetric.euclidean);
          blackhole(res);
          res.dispose();
        });

        c.bench('pdist Cosine [shape=500x100]', () {
          final res = pdist(distMatA, metric: DistanceMetric.cosine);
          blackhole(res);
          res.dispose();
        });

        c.bench('pdist Hamming [shape=500x100 int32]', () {
          final res = pdist(distIntA, metric: DistanceMetric.hamming);
          blackhole(res);
          res.dispose();
        });

        c.bench('cdist Euclidean [shape=500x100 vs 500x100]', () {
          final res = cdist(
            distMatA,
            distMatB,
            metric: DistanceMetric.euclidean,
          );
          blackhole(res);
          res.dispose();
        });

        c.bench('cdist Cosine [shape=500x100 vs 500x100]', () {
          final res = cdist(distMatA, distMatB, metric: DistanceMetric.cosine);
          blackhole(res);
          res.dispose();
        });

        c.bench('cdist Hamming [shape=500x100 vs 500x100 int32]', () {
          final res = cdist(distIntA, distIntB, metric: DistanceMetric.hamming);
          blackhole(res);
          res.dispose();
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/master',
    ),
  );
}
