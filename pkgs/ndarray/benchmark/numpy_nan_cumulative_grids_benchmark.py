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

from numpy.lib.stride_tricks import sliding_window_view
from numpy_bench_helper import NumpyBenchSuite, np


def main():
    size = 100000
    dim = 500

    suite = NumpyBenchSuite(
        "NumPy NaN Reductions, Cumulative Scans, Floating-Point & Grids Benchmark Suite",
        "nan_cumulative_grids",
    )

    nan_vec = np.array(
        [np.nan if i % 10 == 0 else float((i % 100) + 1) for i in range(size)],
        dtype=np.float64,
    )
    clean_vec = np.linspace(1.00001, 1.00002, size, dtype=np.float64)
    clean_vec_jitter = np.linspace(
        1.000010001, 1.000020001, size, dtype=np.float64
    )
    mat2d = np.linspace(0.0, 100.0, dim * dim, dtype=np.float64).reshape(
        (dim, dim)
    )

    suite.group("1. NaN-Resilient Statistical Reductions (10% NaN)")
    suite.bench(
        "nansum(arr) [100k Float64]",
        lambda: np.nansum(nan_vec),
        iterations=300,
    )
    suite.bench(
        "nanmean(arr) [100k Float64]",
        lambda: np.nanmean(nan_vec),
        iterations=300,
    )
    suite.bench(
        "nanstd(arr) [100k Float64]",
        lambda: np.nanstd(nan_vec),
        iterations=200,
    )
    suite.bench(
        "nanvar(arr) [100k Float64]",
        lambda: np.nanvar(nan_vec),
        iterations=200,
    )
    suite.bench(
        "nanmin(arr) [100k Float64]",
        lambda: np.nanmin(nan_vec),
        iterations=300,
    )
    suite.bench(
        "nanmax(arr) [100k Float64]",
        lambda: np.nanmax(nan_vec),
        iterations=300,
    )

    suite.group("2. Cumulative Scans (cumsum & cumprod)")
    suite.bench(
        "cumsum(arr) [100k Float64]",
        lambda: np.cumsum(clean_vec),
        iterations=300,
    )
    suite.bench(
        "cumsum(mat, axis=0) [500x500 Float64]",
        lambda: np.cumsum(mat2d, axis=0),
        iterations=200,
    )
    suite.bench(
        "cumsum(mat, axis=1) [500x500 Float64]",
        lambda: np.cumsum(mat2d, axis=1),
        iterations=200,
    )
    suite.bench(
        "cumprod(arr) [100k Float64]",
        lambda: np.cumprod(clean_vec),
        iterations=300,
    )

    suite.group("3. Floating-Point Inspection, Tolerances & Mesh Grids")
    suite.bench(
        "isnan(arr) [100k Float64]",
        lambda: np.isnan(nan_vec),
        iterations=500,
    )
    suite.bench(
        "isfinite(arr) [100k Float64]",
        lambda: np.isfinite(nan_vec),
        iterations=500,
    )
    suite.bench(
        "isClose(a, b) [100k Float64]",
        lambda: np.isclose(clean_vec, clean_vec_jitter),
        iterations=200,
    )
    suite.bench(
        "allClose(a, b) [100k Float64]",
        lambda: np.allclose(clean_vec, clean_vec_jitter),
        iterations=200,
    )
    suite.bench(
        "copysign(a, b) [100k Float64]",
        lambda: np.copysign(clean_vec, nan_vec),
        iterations=400,
    )
    suite.bench(
        "mgrid([0:500, 0:500]) [2x500x500 dense grid]",
        lambda: np.mgrid[0:500:1.0, 0:500:1.0],
        iterations=150,
    )

    row_vec = np.linspace(0.0, 10.0, dim, dtype=np.float64)
    suite.bench(
        "broadcastTo(vec, [500, 500]) [zero-copy view]",
        lambda: np.broadcast_to(row_vec, (dim, dim)),
        iterations=1000,
    )
    suite.bench(
        "slidingWindowView(arr, [16]) [100k 1D window view]",
        lambda: sliding_window_view(clean_vec, (16,)),
        iterations=1000,
    )
    suite.bench(
        "tril(mat) [500x500]",
        lambda: np.tril(mat2d),
        iterations=300,
    )
    suite.bench(
        "triu(mat) [500x500]",
        lambda: np.triu(mat2d),
        iterations=300,
    )

    suite.finish()


if __name__ == "__main__":
    main()
