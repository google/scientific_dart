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
    size1d = 1000000
    dim2d = 1000

    suite = NumpyBenchSuite(
        "NumPy (Python) Ufunc Reductions & Masked Functions Benchmark",
        "ufunc_reductions",
    )

    a1d = (np.arange(size1d, dtype=np.float64) % 100.0).astype(np.float64)
    b1d = ((np.arange(size1d, dtype=np.float64) % 50.0) + 1.0).astype(np.float64)
    mask1d = (np.arange(size1d) % 2 == 0)

    a2d = (np.arange(dim2d * dim2d, dtype=np.float64) % 100.0).reshape(
        (dim2d, dim2d)
    )
    b2d = ((np.arange(dim2d * dim2d, dtype=np.float64) % 50.0) + 1.0).reshape(
        (dim2d, dim2d)
    )

    a_outer = np.arange(1000, dtype=np.float64)
    b_outer = np.arange(1, 1001, dtype=np.float64)

    indices_at = ((np.arange(10000, dtype=np.int64) * 97) % size1d).astype(
        np.int64
    )
    vals_at = np.arange(10000, dtype=np.float64)
    indices_reduceat = (np.arange(1000, dtype=np.int64) * 1000).astype(np.int64)

    out1d = np.zeros(size1d, dtype=np.float64)

    suite.group("1. Reductions")
    suite.bench(
        "reduce(add) [1D 1M global]",
        lambda: np.add.reduce(a1d),
        iterations=100,
    )
    suite.bench(
        "reduce(add) [2D 1000x1000 axis:0]",
        lambda: np.add.reduce(a2d, axis=0),
        iterations=100,
    )
    suite.bench(
        "reduce(multiply) [1D 100K global]",
        lambda: np.multiply.reduce(a1d[:100000]),
        iterations=100,
    )

    suite.group("2. Accumulate")
    suite.bench(
        "accumulate(add) [1D 1M cumsum]",
        lambda: np.add.accumulate(a1d),
        iterations=100,
    )
    suite.bench(
        "accumulate(add) [2D 1000x1000 axis:0]",
        lambda: np.add.accumulate(a2d, axis=0),
        iterations=100,
    )

    suite.group("3. Outer Product")
    suite.bench(
        "outer(add) [1000 x 1000]",
        lambda: np.add.outer(a_outer, b_outer),
        iterations=100,
    )
    suite.bench(
        "outer(multiply) [1000 x 1000]",
        lambda: np.multiply.outer(a_outer, b_outer),
        iterations=100,
    )

    suite.group("4. Segment Reduction & Scatter (reduceat & at)")
    suite.bench(
        "reduceat(add) [1M array, 1000 segments]",
        lambda: np.add.reduceat(a1d, indices_reduceat),
        iterations=100,
    )
    suite.bench(
        "at(add) [1M array, 10K scatter updates]",
        lambda: np.add.at(a1d, indices_at, vals_at),
        iterations=100,
    )

    suite.group("5. Masked Functions (where=)")
    suite.bench(
        "add(where=mask) [1D 1M contiguous]",
        lambda: np.add(a1d, b1d, where=mask1d, out=out1d),
        iterations=100,
    )
    suite.bench(
        "multiply(where=mask) [1D 1M contiguous]",
        lambda: np.multiply(a1d, b1d, where=mask1d, out=out1d),
        iterations=100,
    )

    a_strided = a2d.ravel()[::2]
    b_strided = b2d.ravel()[::2]
    out_strided = np.zeros(a_strided.shape[0], dtype=np.float64)
    suite.bench(
        "add(where=mask) [1D strided view step=2]",
        lambda: np.add(a_strided, b_strided, out=out_strided),
        iterations=100,
    )

    suite.finish()


if __name__ == "__main__":
    main()
