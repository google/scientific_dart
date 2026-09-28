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

import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    // 1. Future Value (fv)
    // What is the future value after 10 years of saving $100 now, with
    // an additional monthly savings of $100. Assume the interest rate is
    // 5% (annually) compounded monthly?
    final NDArray<Float64> rate = NDArray<Float64>.scalar(
      0.05 / 12,
      dtype: DType.float64,
    );
    final NDArray<Float64> nper = NDArray<Float64>.scalar(
      10.0 * 12,
      dtype: DType.float64,
    );
    final NDArray<Float64> pmt = NDArray<Float64>.scalar(
      -100.0,
      dtype: DType.float64,
    );
    final NDArray<Float64> pvVal = NDArray<Float64>.scalar(
      -100.0,
      dtype: DType.float64,
    );

    final NDArray<Float64> fvResult = fv(rate, nper, pmt, pvVal);
    print(
      'FV Result: \$${fvResult.scalar.toStringAsFixed(2)}',
    ); // Expected: $15692.93

    // 2. Present Value (pv)
    // What is the present value (investment) needed to get $15,692.93 after
    // 10 years of saving $100/month at 5% annual interest?
    final NDArray<Float64> fvVal = NDArray<Float64>.scalar(
      15692.92889433575,
      dtype: DType.float64,
    );
    final NDArray<Float64> pvResult = pv(rate, nper, pmt, fvVal);
    print(
      'PV Result: \$${pvResult.scalar.toStringAsFixed(2)}',
    ); // Expected: -$100.00

    // 3. Net Present Value (npv)
    // Cash flows: invest 100, then withdraw 39, 59, 55, 20
    final NDArray<Float64> cashFlows = NDArray.fromList(
      <double>[-100.0, 39.0, 59.0, 55.0, 20.0],
      [5],
      DType.float64,
    );
    final NDArray<Float64> discountRate = NDArray<Float64>.scalar(
      0.08,
      dtype: DType.float64,
    );
    final NDArray<Float64> npvResult = npv(discountRate, cashFlows);
    print('NPV Result: \$${npvResult.scalar.toStringAsFixed(2)}');

    // 4. Internal Rate of Return (irr)
    // Solves for rate where NPV is 0.
    final NDArray<Float64> irrResult = irr(cashFlows);
    print(
      'IRR Result: ${(irrResult.scalar * 100).toStringAsFixed(2)}%',
    ); // Expected: 28.10%
  });
}
