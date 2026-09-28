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

import 'dart:math' as math;

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);
  const size = 100000;

  await criterion(
    'NDArray Binning, Histograms & Set Operations Benchmark Suite',
    (c) {
      final rand = math.Random(42);
      final rawData = NDArray<double>.fromList(
        List.generate(size, (_) => rand.nextDouble() * 100.0),
        [size],
        DType.float64,
      );

      c.group('1. Binning & Histograms', () {
        c.bench('histogram(data, bins: 100) [size=100,000]', () {
          final res = histogram(rawData, bins: 100);
          blackhole(res.hist);
          res.hist.dispose();
          res.binEdges.dispose();
        }, throughput: Throughput.elements(size));

        final intData = NDArray<int>.fromList(
          List.generate(size, (_) => rand.nextInt(500)),
          [size],
          DType.int32,
        );

        c.bench('bincount(intData) [size=100,000, 500 bins]', () {
          final counts = bincount(intData);
          blackhole(counts);
          counts.dispose();
        }, throughput: Throughput.elements(size));

        final bins = linspace<double>(0.0, 100.0, 101, dtype: DType.float64);
        c.bench('digitize(data, bins: 100) [size=100,000]', () {
          final binIdx = digitize(rawData, bins);
          blackhole(binIdx);
          binIdx.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Set Operations', () {
        final repeatedData = NDArray<int>.fromList(
          List.generate(size, (_) => rand.nextInt(10000)),
          [size],
          DType.int32,
        );

        c.bench('unique(data) [size=100,000, ~10,000 unique]', () {
          final u = unique(repeatedData);
          blackhole(u);
          u.dispose();
        }, throughput: Throughput.elements(size));

        final setA = NDArray<int>.fromList(
          List.generate(50000, (_) => rand.nextInt(50000)),
          [50000],
          DType.int32,
        );
        final setB = NDArray<int>.fromList(
          List.generate(50000, (_) => rand.nextInt(50000)),
          [50000],
          DType.int32,
        );

        c.bench('isin(setA, setB) [50,000 vs 50,000]', () {
          final mask = isin(setA, setB);
          blackhole(mask);
          mask.dispose();
        }, throughput: Throughput.elements(50000));

        c.bench('intersect1d(setA, setB) [50,000 vs 50,000]', () {
          final inter = intersect1d(setA, setB);
          blackhole(inter);
          inter.dispose();
        }, throughput: Throughput.elements(50000));

        c.bench('union1d(setA, setB) [50,000 vs 50,000]', () {
          final u = union1d(setA, setB);
          blackhole(u);
          u.dispose();
        }, throughput: Throughput.elements(100000));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/binning_set_ops',
    ),
  );
}
