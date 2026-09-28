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

import numpy.polynomial.chebyshev as cheb
from numpy_bench_helper import NumpyBenchSuite, np


def main():
    size = 100000
    rng = np.random.default_rng(42)

    suite = NumpyBenchSuite(
        "NumPy Polynomial Fitting & 1D Interpolation Benchmark Suite",
        "polynomial_interp",
    )

    x_points = np.linspace(-10.0, 10.0, size, dtype=np.float64)
    coeffs5 = np.array([1.0, -2.5, 0.4, 3.2, -1.1, 0.5], dtype=np.float64)

    suite.group("1. Polynomial Evaluation & Orthogonal Series")
    suite.bench(
        "polyval(deg=5, x) [size=100,000]",
        lambda: np.polyval(coeffs5, x_points),
        iterations=200,
    )

    cheb_coeffs = np.array([0.5, 1.2, -0.8, 2.1, 0.3], dtype=np.float64)
    x_norm = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    suite.bench(
        "chebval(deg=4, x) [size=100,000]",
        lambda: cheb.chebval(x_norm, cheb_coeffs),
        iterations=200,
    )

    suite.group("2. Least-Squares Polynomial Fitting (polyfit)")
    fit_n = 10000
    x_fit = np.linspace(0.0, 10.0, fit_n, dtype=np.float64)
    idx = np.arange(fit_n, dtype=np.float64)
    y_fit = (idx * 0.1) ** 2 + rng.random(fit_n, dtype=np.float64) * 0.5

    suite.bench(
        "polyfit(x, y, deg=3) [N=10,000 points]",
        lambda: np.polyfit(x_fit, y_fit, 3),
        iterations=200,
    )
    suite.bench(
        "polyfit(x, y, deg=9) [N=10,000 points]",
        lambda: np.polyfit(x_fit, y_fit, 9),
        iterations=200,
    )

    suite.group("3. 1D Piecewise Linear Interpolation")
    num_knots = 1000
    xp = np.linspace(0.0, 100.0, num_knots, dtype=np.float64)
    fp = np.sin(np.arange(num_knots, dtype=np.float64) * 0.1)
    x_query = np.linspace(0.0, 100.0, size, dtype=np.float64)

    suite.bench(
        "interp(xQuery, xp, fp) [100,000 queries across 1,000 knots]",
        lambda: np.interp(x_query, xp, fp),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()
