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
  const nSims = 10000;

  await criterion(
    'NDArray Quantitative Financial Operations Benchmark Suite',
    (c) {
      final rate = linspace(0.01, 0.15, nSims, dtype: DType.float64);
      final nper = linspace(1.0, 30.0, nSims, dtype: DType.float64);
      final pmt = linspace(-1000.0, -100.0, nSims, dtype: DType.float64);
      final pvVal = linspace(10000.0, 100000.0, nSims, dtype: DType.float64);
      final fvVal = linspace(0.0, 50000.0, nSims, dtype: DType.float64);

      c.group('1. Time Value of Money (10k parameter simulations)', () {
        c.bench('fv(rate, nper, pmt, pv) [10k]', () {
          final res = fv(rate, nper, pmt, pvVal);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(nSims));

        c.bench('pv(rate, nper, pmt, fv) [10k]', () {
          final res = pv(rate, nper, pmt, fvVal);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(nSims));
      });

      c.group('2. Cash Flow Discounting & Returns', () {
        const nPeriods = 10000;
        final singleRate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
        final cashFlows = linspace(
          -1000.0,
          500.0,
          nPeriods,
          dtype: DType.float64,
        );

        c.bench('npv(rate=0.05, cashflows=[10k])', () {
          final res = npv(singleRate, cashFlows);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(nPeriods));

        final irrFlows = NDArray<Float64>.fromList(
          [-10000.0, ...List.filled(49, 350.0)],
          [50],
          DType.float64,
        );

        c.bench('irr(cashflows=[50 periods])', () {
          final res = irr(irrFlows);
          blackhole(res);
          res.dispose();
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/financial',
    ),
  );
}
