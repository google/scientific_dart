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
    matrix_dim = 500

    suite = NumpyBenchSuite(
        "NumPy Indexing, Slicing & Manipulation Benchmark Suite",
        "indexing_manipulation",
    )

    a1d = np.linspace(0.0, 100.0, size, dtype=np.float64)
    mat2d = np.arange(matrix_dim * matrix_dim, dtype=np.float64).reshape(
        (matrix_dim, matrix_dim)
    )

    suite.group("1. Views, Slicing & Reshaping (Zero-Copy Metadata)")
    suite.bench(
        "1D Strided Slice a[::2] [size=100,000]",
        lambda: a1d[::2],
        iterations=1000,
    )
    suite.bench(
        "2D Multi-Axis Strided Slice mat[10:490:2, 20:480:3] [500x500]",
        lambda: mat2d[10:490:2, 20:480:3],
        iterations=1000,
    )
    suite.bench(
        "2D Transpose (strides swap) [500x500]",
        lambda: mat2d.T,
        iterations=1000,
    )
    suite.bench(
        "Reshape [500, 500] -> [250, 1000]",
        lambda: mat2d.reshape((250, 1000)),
        iterations=1000,
    )

    suite.group("2. Advanced Indexing & Selection")
    indices = np.arange(matrix_dim, dtype=np.int32).reshape((matrix_dim, 1))
    suite.bench(
        "take_along_axis [500x500, axis=0]",
        lambda: np.take_along_axis(mat2d, indices, axis=0),
        iterations=500,
    )

    put_values = np.ones((matrix_dim, 1), dtype=np.float64)
    suite.bench(
        "put_along_axis [500x500, axis=0]",
        lambda: np.put_along_axis(mat2d, indices, put_values, axis=0),
        iterations=500,
    )

    suite.bench(
        "diag() Extraction [500x500]",
        lambda: np.diag(mat2d).copy(),
        iterations=500,
    )

    choices = [
        np.zeros(10000, dtype=np.float64),
        np.ones(10000, dtype=np.float64),
        np.linspace(0.0, 10.0, 10000, dtype=np.float64),
    ]
    selector = np.array([i % 3 for i in range(10000)], dtype=np.int32)
    suite.bench(
        "choose() Multi-Array Selection [size=10,000, 3 choices]",
        lambda: np.choose(selector, choices),
        iterations=500,
    )

    suite.group("3. Array Assembly & Joining")
    block_a = np.ones((250, 500), dtype=np.float64)
    block_b = np.zeros((250, 500), dtype=np.float64)

    suite.bench(
        "concatenate([A, B], axis=0) [250x500 + 250x500 -> 500x500]",
        lambda: np.concatenate([block_a, block_b], axis=0),
        iterations=300,
    )
    suite.bench(
        "stack([A, B], axis=0) [2 x 250x500 -> 2x250x500]",
        lambda: np.stack([block_a, block_b], axis=0),
        iterations=300,
    )

    small_tile = np.ones((50, 50), dtype=np.float64)
    suite.bench(
        "tile([50, 50], [10, 10]) -> [500, 500]",
        lambda: np.tile(small_tile, (10, 10)),
        iterations=300,
    )

    rep_vec = np.linspace(0.0, 10.0, 1000, dtype=np.float64)
    suite.bench(
        "repeat([1000], repeats=50) -> [50,000]",
        lambda: np.repeat(rep_vec, 50),
        iterations=500,
    )

    suite.finish()


if __name__ == "__main__":
    main()
