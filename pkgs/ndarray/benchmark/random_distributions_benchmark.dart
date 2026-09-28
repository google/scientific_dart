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

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);
  const size = 100000;

  await criterion(
    'NDArray Random Number Generation & Distributions Benchmark Suite',
    (c) {
      c.group('1. Continuous & Discrete Distributions (100k samples)', () {
        c.bench('uniform([100k])', () {
          final res = uniform<DTypeTag>([size]);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('normal([100k], loc=5.0, scale=2.0)', () {
          final res = normal<DTypeTag>([size], loc: 5.0, scale: 2.0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('exponential([100k], scale=1.5)', () {
          final res = exponential<DTypeTag>([size], scale: 1.5);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('poisson([100k], lam=5.0)', () {
          final res = poisson<DTypeTag>([size], lam: 5.0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('binomial([100k], n=10, p=0.5)', () {
          final res = binomial<DTypeTag>([size], n: 10, p: 0.5);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('randint([100k], low=0, high=100)', () {
          final res = randint<DTypeTag>([size], low: 0, high: 100);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Permutations, Choice & Shuffling', () {
        final samplePool = linspace<DTypeTag>(
          0.0,
          100.0,
          size,
          dtype: DType.float64,
        );

        c.bench('choice(pool, size=100k, replace=true)', () {
          final res = choice(samplePool, size: [size], replace: true);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('permutation(arr) [100k]', () {
          final res = permutation(samplePool);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        final shuffleArr = linspace<DTypeTag>(
          0.0,
          100.0,
          size,
          dtype: DType.float64,
        );
        c.bench('shuffle(arr) [100k in-place]', () {
          shuffle(shuffleArr);
          blackhole(shuffleArr);
        }, throughput: Throughput.elements(size));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/random_distributions',
    ),
  );
}
