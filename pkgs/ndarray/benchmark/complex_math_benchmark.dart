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
    'NDArray Complex Number Vectorized Math Benchmark Suite',
    (c) {
      final cA = NDArray<AnySpec>.fromList(
        List.generate(
          size,
          (i) => Complex(i.toDouble() * 0.01, (size - i).toDouble() * 0.01),
        ),
        [size],
        DType.complex128,
      );

      final cB = NDArray<AnySpec>.fromList(
        List.generate(size, (i) => Complex((i % 100).toDouble() + 1.0, -0.5)),
        [size],
        DType.complex128,
      );

      c.group('1. Vectorized Complex Arithmetic (Complex128)', () {
        c.bench('cMul (cA * cB) [size=100,000 Complex128]', () {
          final res = multiply(cA, cB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('cDiv (cA / cB) [size=100,000 Complex128]', () {
          final res = divide(cA, cB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('cAdd (cA + cB) [size=100,000 Complex128]', () {
          final res = add(cA, cB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Complex Transformations & Projections', () {
        c.bench('conj(cA) [size=100,000 Complex128]', () {
          final res = conj(cA);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('abs(cA) (Magnitude) [size=100,000 Complex128 -> Float64]', () {
          final res = abs(cA);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('angle(cA) (Phase) [size=100,000 Complex128 -> Float64]', () {
          final res = angle(cA);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('exp(cA) (Complex Exponential) [size=100,000 Complex128]', () {
          final res = exp(cA);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/complex_math',
    ),
  );
}
