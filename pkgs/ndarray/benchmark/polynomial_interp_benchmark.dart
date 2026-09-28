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
    'NDArray Polynomial Fitting & 1D Interpolation Benchmark Suite',
    (c) {
      final rand = math.Random(42);
      final xPoints = linspace<DTypeTag>(
        -10.0,
        10.0,
        size,
        dtype: DType.float64,
      );
      final coeffs5 = NDArray<DTypeTag>.fromList(
        [1.0, -2.5, 0.4, 3.2, -1.1, 0.5],
        [6],
        DType.float64,
      );

      c.group('1. Polynomial Evaluation & Orthogonal Series', () {
        c.bench('polyval(deg=5, x) [size=100,000]', () {
          final y = polyval(coeffs5, xPoints);
          blackhole(y);
          y.dispose();
        }, throughput: Throughput.elements(size));

        final chebCoeffs = NDArray<DTypeTag>.fromList(
          [0.5, 1.2, -0.8, 2.1, 0.3],
          [5],
          DType.float64,
        );
        final xNorm = linspace<DTypeTag>(-1.0, 1.0, size, dtype: DType.float64);

        c.bench('chebval(deg=4, x) [size=100,000]', () {
          final y = chebval(xNorm, chebCoeffs);
          blackhole(y);
          y.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Least-Squares Polynomial Fitting (polyfit)', () {
        const fitN = 10000;
        final xFit = linspace<DTypeTag>(0.0, 10.0, fitN, dtype: DType.float64);
        final yFit = NDArray<DTypeTag>.fromList(
          List.generate(
            fitN,
            (i) => (i * 0.1) * (i * 0.1) + rand.nextDouble() * 0.5,
          ),
          [fitN],
          DType.float64,
        );

        c.bench('polyfit(x, y, deg=3) [N=10,000 points]', () {
          final fit = polyfit(xFit, yFit, 3);
          blackhole(fit);
          fit.dispose();
        }, throughput: Throughput.elements(fitN));

        c.bench('polyfit(x, y, deg=9) [N=10,000 points]', () {
          final fit = polyfit(xFit, yFit, 9);
          blackhole(fit);
          fit.dispose();
        }, throughput: Throughput.elements(fitN));
      });

      c.group('3. 1D Piecewise Linear Interpolation', () {
        const numKnots = 1000;
        final xp = linspace<DTypeTag>(
          0.0,
          100.0,
          numKnots,
          dtype: DType.float64,
        );
        final fp = NDArray<DTypeTag>.fromList(
          List.generate(numKnots, (i) => math.sin(i * 0.1)),
          [numKnots],
          DType.float64,
        );
        final xQuery = linspace<DTypeTag>(
          0.0,
          100.0,
          size,
          dtype: DType.float64,
        );

        c.bench(
          'interp(xQuery, xp, fp) [100,000 queries across 1,000 knots]',
          () {
            final yInterp = interp(xQuery, xp, fp);
            blackhole(yInterp);
            yInterp.dispose();
          },
          throughput: Throughput.elements(size),
        );
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/polynomial_interp',
    ),
  );
}
