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
  await criterion(
    'NDArray Calculus Benchmarks',
    (c) {
      final y1d = NDArray.zeros([1000000], DType.float64);
      for (var i = 0; i < 1000000; i++) {
        y1d.data[i] = i.toDouble();
      }

      final f1d = NDArray.zeros([1000000], DType.float64);
      for (var i = 0; i < 1000000; i++) {
        f1d.data[i] = i.toDouble() * i.toDouble();
      }

      final f2d = NDArray.zeros([1000, 1000], DType.float64);
      for (var i = 0; i < 1000000; i++) {
        f2d.data[i] = i.toDouble();
      }

      c.bench('Calculus | trapz 1D (Float64) [size=1,000,000]', () {
        final res = trapz(y1d);
        blackhole(res);
        res.dispose();
      }, throughput: Throughput.elements(1000000));

      c.bench('Calculus | gradient 1D (Float64) [size=1,000,000]', () {
        final res = gradient(f1d);
        blackhole(res);
        res.dispose();
      }, throughput: Throughput.elements(1000000));

      c.bench('Calculus | gradient 2D (Float64) [size=1,000x1,000]', () {
        final res = gradient(f2d, axis: 0);
        blackhole(res);
        res.dispose();
      }, throughput: Throughput.elements(1000000));
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/calculus',
    ),
  );
}
