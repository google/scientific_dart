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


def main():
    size = 100000
    matrix_rows = 1000
    matrix_cols = 1000

    suite = NumpyBenchSuite(
        "NumPy Statistics & Reductions Benchmark Suite",
        "stats_reductions",
    )

    vec1d = np.linspace(0.0, 100.0, size, dtype=np.float64)
    mat2d = np.linspace(
        0.0, 1000.0, matrix_rows * matrix_cols, dtype=np.float64
    ).reshape((matrix_rows, matrix_cols))

    suite.group("1. Basic Reductions & Moments")
    suite.bench("mean() [1D flat 100k]", lambda: np.mean(vec1d), iterations=500)
    suite.bench(
        "mean(mat, axis=0) [1000x1000]",
        lambda: np.mean(mat2d, axis=0),
        iterations=200,
    )
    suite.bench(
        "mean(mat, axis=1) [1000x1000]",
        lambda: np.mean(mat2d, axis=1),
        iterations=200,
    )
    suite.bench("std() [1D flat 100k]", lambda: np.std(vec1d), iterations=300)
    suite.bench("var_() [1D flat 100k]", lambda: np.var(vec1d), iterations=300)

    weights = np.linspace(1.0, 10.0, size, dtype=np.float64)
    suite.bench(
        "average(vec, weights=w) [100k]",
        lambda: np.average(vec1d, weights=weights),
        iterations=300,
    )

    suite.group("2. Order Statistics (Median & Quantile)")
    rand_vec = np.array(
        [((i * 37) % 1000) * 1.0 for i in range(size)], dtype=np.float64
    )
    suite.bench(
        "median() [100k items]", lambda: np.median(rand_vec), iterations=100
    )
    suite.bench(
        "quantile(p=0.75) [100k items]",
        lambda: np.quantile(rand_vec, 0.75),
        iterations=100,
    )
    suite.bench(
        "ptp() (Peak-to-Peak) [100k items]",
        lambda: np.ptp(rand_vec),
        iterations=500,
    )

    suite.group("3. Covariance & Correlation")
    n_vars = 50
    n_obs = 500
    obs_mat = np.linspace(
        0.0, 100.0, n_vars * n_obs, dtype=np.float64
    ).reshape((n_vars, n_obs))

    suite.bench(
        "cov(X) [50 vars x 500 obs]", lambda: np.cov(obs_mat), iterations=300
    )
    suite.bench(
        "corrcoef(X) [50 vars x 500 obs]",
        lambda: np.corrcoef(obs_mat),
        iterations=300,
    )

    suite.finish()


if __name__ == "__main__":
    main()
