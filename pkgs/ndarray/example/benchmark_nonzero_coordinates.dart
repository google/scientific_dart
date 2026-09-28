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
  const size = 80; // 80 x 80 matrix

  await criterion(
    'NDArray Non-Zero Coordinates Search Benchmark',
    (c) {
      final a = NDArray.zeros([size, size], DType.float64);
      for (var i = 0; i < size; i += 5) {
        for (var j = 0; j < size; j += 3) {
          a.setCell([i, j], Float64(9.9));
        }
      }

      c.group('nonzero() Search Strategy', () {
        c.bench('1. Optimized Flat Offset nonzero()', () {
          final res = nonzero(a);
          blackhole(res.length);
          for (final list in res) {
            list.dispose();
          }
        }, throughput: Throughput.elements(size * size));

        c.bench('2. Slow Bracket Selector Sweep Fallback (a[[i, j]])', () {
          final coordinateLists = List.generate(2, (_) => <int>[]);
          for (var i = 0; i < size; i++) {
            for (var j = 0; j < size; j++) {
              final val = a[[i, j]];
              if (val != 0.0) {
                coordinateLists[0].add(i);
                coordinateLists[1].add(j);
              }
            }
          }
          final res = coordinateLists.map((list) {
            return NDArray<int>.fromList(list, [list.length], DType.int32);
          }).toList();
          blackhole(res.length);
          for (final list in res) {
            list.dispose();
          }
        }, throughput: Throughput.elements(size * size));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/nonzero_coords',
    ),
  );
}
