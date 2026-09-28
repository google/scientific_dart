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
    dim = 500
    mat = np.linspace(0.0, 100.0, dim * dim, dtype=np.float64).reshape(
        (dim, dim)
    )

    suite = NumpyBenchSuite(
        "NumPy Padding, Rotations, Rolling & Splitting Benchmark Suite",
        "padding_transforms",
    )

    suite.group("1. Multidimensional Array Padding")
    suite.bench(
        "pad(constant, pad_width=10) [500x500 -> 520x520]",
        lambda: np.pad(mat, 10, mode="constant"),
        iterations=200,
    )
    suite.bench(
        "pad(edge, pad_width=10) [500x500 -> 520x520]",
        lambda: np.pad(mat, 10, mode="edge"),
        iterations=200,
    )
    suite.bench(
        "pad(reflect, pad_width=10) [500x500 -> 520x520]",
        lambda: np.pad(mat, 10, mode="reflect"),
        iterations=200,
    )

    suite.group("2. Array Flipping & Rolling")
    suite.bench(
        "roll([20, 20]) [500x500]",
        lambda: np.roll(mat, (20, 20), axis=(0, 1)),
        iterations=200,
    )
    suite.bench(
        "flip(axis=0) [500x500]",
        lambda: np.flip(mat, axis=0),
        iterations=1000,
    )
    suite.bench(
        "fliplr() [500x500]",
        lambda: np.fliplr(mat),
        iterations=1000,
    )

    suite.group("3. Array Splitting & Chunking")
    large_rows = 1000
    large_mat = np.linspace(
        0.0, 100.0, large_rows * dim, dtype=np.float64
    ).reshape((large_rows, dim))

    suite.bench(
        "split(10 chunks, axis=0) [1000x500]",
        lambda: np.split(large_mat, 10, axis=0),
        iterations=1000,
    )

    suite.finish()


if __name__ == "__main__":
    main()
