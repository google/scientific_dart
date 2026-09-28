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
    'NDArray Non-Contiguous sin() Performance Benchmark',
    (c) {
      final mat = NDArray<double>.zeros([2000, 2000], DType.float64);
      for (var i = 0; i < mat.data.length; i++) {
        mat.data[i] = i.toDouble() / 10000.0;
      }
      final matT = mat.transpose();
      final out = NDArray<double>.create([2000, 2000], DType.float64);

      c.bench('strided sin(matT) [shape=2000x2000 transposed]', () {
        sin(matT, out: out);
      }, throughput: Throughput.elements(2000 * 2000));
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/strided_trig',
    ),
  );
}
