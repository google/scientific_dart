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
import 'package:ndarray/ndarray.dart' as nd;
import 'package:ndarray_ma/ndarray_ma.dart';

void main() async {
  nd.setNumThreads(1);
  const size = 100000;

  await criterion(
    'MaskedArray (ndarray_ma) Performance Benchmarks',
    (c) {
      final rand = math.Random(42);
      final rawData1 = nd.NDArray<nd.Float64>.fromList(
        List.generate(size, (_) => rand.nextDouble() * 100.0),
        [size],
        nd.DType.float64,
      );
      final rawData2 = nd.NDArray<nd.Float64>.fromList(
        List.generate(size, (_) => rand.nextDouble() * 100.0),
        [size],
        nd.DType.float64,
      );

      // Create boolean mask with ~20% masked values
      final maskData = List.generate(size, (_) => rand.nextDouble() < 0.2);
      final mask = nd.NDArray.fromList(maskData, [size], nd.DType.boolean);

      final ma1 = MaskedArray(rawData1, mask);
      final ma2 = MaskedArray(rawData2, mask);

      c.group('1. Arithmetic: Masked vs Unmasked', () {
        final outRaw = nd.NDArray.create([size], nd.DType.float64);

        c.bench('Raw NDArray add (out buffer) [$size]', () {
          nd.add(rawData1, rawData2, out: outRaw);
        }, throughput: Throughput.elements(size));

        c.bench('MaskedArray add (ma1 + ma2) [$size]', () {
          final res = add(ma1, ma2);
          blackhole(res);
          res.data.dispose();
          res.mask.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('MaskedArray multiply (ma1 * ma2) [$size]', () {
          final res = multiply(ma1, ma2);
          blackhole(res);
          res.data.dispose();
          res.mask.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Masked Reductions', () {
        c.bench('MaskedArray sum() [$size]', () {
          final res = sum(ma1);
          blackhole(res);
          res.data.dispose();
          res.mask.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('MaskedArray mean() [$size]', () {
          final res = mean(ma1);
          blackhole(res);
          res.data.dispose();
          res.mask.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('MaskedArray std() [$size]', () {
          final res = std(ma1);
          blackhole(res);
          res.data.dispose();
          res.mask.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('3. Masking & Compression Utilities', () {
        // Data with NaNs and infinities
        final dataWithInvalid = nd.NDArray<nd.Float64>.fromList(
          List.generate(size, (i) {
            if (i % 10 == 0) return double.nan;
            if (i % 15 == 0) return double.infinity;
            return i.toDouble();
          }),
          [size],
          nd.DType.float64,
        );

        c.bench('MaskedArray.maskedInvalid() [$size]', () {
          final res = MaskedArray.maskedInvalid(dataWithInvalid);
          blackhole(res);
          res.mask.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('MaskedArray.maskedGreater() [$size]', () {
          final res = MaskedArray.maskedGreater(rawData1, 50.0);
          blackhole(res);
          res.mask.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('compressed() (Extract valid elements) [$size]', () {
          final res = ma1.compressed();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report',
    ),
  );
}
