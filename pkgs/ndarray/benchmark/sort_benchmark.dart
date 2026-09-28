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
import 'dart:typed_data';

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);

  final sizes = [1000, 10000, 50000];

  await criterion(
    'NDArray Timsort & Argsort Comprehensive Benchmark Suite',
    (c) {
      void registerTrack(
        String label,
        Float64List Function(int size) templateGen,
      ) {
        c.group(label, () {
          for (final size in sizes) {
            final template = templateGen(size);

            c.bench<NDArray<double>>(
              'Direct sort() [$size]',
              (arr) {
                final res = sort(arr);
                blackhole(res);
                res.dispose();
                arr.dispose();
              },
              setup: () =>
                  NDArray<double>.fromList(template, [size], DType.float64),
              batchSize: BatchSize.largeInput,
              throughput: Throughput.elements(size),
            );

            c.bench<NDArray<double>>(
              'Indirect argsort() [$size]',
              (arr) {
                final res = argsort(arr);
                blackhole(res);
                res.dispose();
                arr.dispose();
              },
              setup: () =>
                  NDArray<double>.fromList(template, [size], DType.float64),
              batchSize: BatchSize.largeInput,
              throughput: Throughput.elements(size),
            );
          }
        });
      }

      registerTrack('Random Array', (size) {
        final rand = math.Random(42);
        return Float64List.fromList(
          List.generate(size, (_) => rand.nextDouble() * 1000.0),
        );
      });

      registerTrack('Already Sorted', (size) {
        return Float64List.fromList(List.generate(size, (i) => i.toDouble()));
      });

      registerTrack('Reverse Sorted', (size) {
        return Float64List.fromList(
          List.generate(size, (i) => (size - i).toDouble()),
        );
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/sort',
    ),
  );
}
