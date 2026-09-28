# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy_bench_helper import NumpyBenchSuite, np


def fv_numpy(rate, nper, pmt, pv):
    temp = (1.0 + rate) ** nper
    return -pv * temp - pmt * (temp - 1.0) / rate


def pv_numpy(rate, nper, pmt, fv):
    temp = (1.0 + rate) ** nper
    return (-fv - pmt * (temp - 1.0) / rate) / temp


def npv_numpy(rate, values):
    t = np.arange(len(values), dtype=np.float64)
    return np.sum(values / ((1.0 + rate) ** t))


def irr_numpy(values):
    # Companion polynomial roots approach matching numpy-financial / ndarray irr
    roots = np.roots(values)
    real_roots = roots[np.isreal(roots)].real
    pos_roots = real_roots[real_roots > 0]
    r = (1.0 / pos_roots) - 1.0
    return r[np.argmin(np.abs(r))]


def main():
    n_sims = 10000

    suite = NumpyBenchSuite(
        "NumPy Quantitative Financial Operations Benchmark Suite",
        "financial",
    )

    rate = np.linspace(0.01, 0.15, n_sims, dtype=np.float64)
    nper = np.linspace(1.0, 30.0, n_sims, dtype=np.float64)
    pmt = np.linspace(-1000.0, -100.0, n_sims, dtype=np.float64)
    pv_val = np.linspace(10000.0, 100000.0, n_sims, dtype=np.float64)
    fv_val = np.linspace(0.0, 50000.0, n_sims, dtype=np.float64)

    suite.group("1. Time Value of Money (10k parameter simulations)")
    suite.bench(
        "fv(rate, nper, pmt, pv) [10k]",
        lambda: fv_numpy(rate, nper, pmt, pv_val),
        iterations=300,
    )
    suite.bench(
        "pv(rate, nper, pmt, fv) [10k]",
        lambda: pv_numpy(rate, nper, pmt, fv_val),
        iterations=300,
    )

    suite.group("2. Cash Flow Discounting & Returns")
    n_periods = 10000
    cash_flows = np.linspace(-1000.0, 500.0, n_periods, dtype=np.float64)
    suite.bench(
        "npv(rate=0.05, cashflows=[10k])",
        lambda: npv_numpy(0.05, cash_flows),
        iterations=300,
    )

    irr_flows = np.array([-10000.0] + [350.0] * 49, dtype=np.float64)
    suite.bench(
        "irr(cashflows=[50 periods])",
        lambda: irr_numpy(irr_flows),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()
