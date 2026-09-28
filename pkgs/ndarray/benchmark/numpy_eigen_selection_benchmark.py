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
    rng = np.random.default_rng(42)

    suite = NumpyBenchSuite(
        "NumPy Eigenvalues, Matrix Chains, Partitioning & Search Benchmark Suite",
        "eigen_selection",
    )

    sym_dim = 100
    raw_sym = (rng.random((sym_dim, sym_dim), dtype=np.float64) - 0.5) * 2.0
    sym_mat = 0.5 * (raw_sym + raw_sym.T) + np.eye(sym_dim, dtype=np.float64) * float(sym_dim)

    gen_dim = 60
    gen_mat = rng.random((gen_dim, gen_dim), dtype=np.float64) - 0.5

    chain_a = np.ones((100, 10), dtype=np.float64)
    chain_b = np.ones((10, 500), dtype=np.float64)
    chain_c = np.ones((500, 20), dtype=np.float64)
    chain_d = np.ones((20, 100), dtype=np.float64)

    inner_a = np.ones((200, 100), dtype=np.float64)
    inner_b = np.ones((200, 100), dtype=np.float64)
    vdot_a = np.linspace(0.0, 10.0, size, dtype=np.float64)
    vdot_b = np.linspace(1.0, 11.0, size, dtype=np.float64)

    suite.group("1. Eigenvalues, Condition Numbers & Matrix Chains")
    suite.bench(
        "eigh(A) [100x100 symmetric]",
        lambda: np.linalg.eigh(sym_mat),
        iterations=150,
    )
    suite.bench(
        "eigvalsh(A) [100x100 symmetric]",
        lambda: np.linalg.eigvalsh(sym_mat),
        iterations=150,
    )
    suite.bench(
        "eig(A) [60x60 general]",
        lambda: np.linalg.eig(gen_mat),
        iterations=100,
    )
    suite.bench(
        "eigvals(A) [60x60 general]",
        lambda: np.linalg.eigvals(gen_mat),
        iterations=100,
    )
    suite.bench(
        "cond(A) [100x100]",
        lambda: np.linalg.cond(sym_mat),
        iterations=150,
    )
    suite.bench(
        "multi_dot([100x10, 10x500, 500x20, 20x100])",
        lambda: np.linalg.multi_dot([chain_a, chain_b, chain_c, chain_d]),
        iterations=200,
    )
    suite.bench(
        "inner(A, B) [200x100, 200x100 -> 200x200]",
        lambda: np.inner(inner_a, inner_b),
        iterations=200,
    )
    suite.bench(
        "vdot(a, b) [100k Float64]",
        lambda: np.vdot(vdot_a, vdot_b),
        iterations=500,
    )

    suite.group("2. Order Selection, Partitioning & Binary Search")
    rand_vec = rng.random(size, dtype=np.float64) * 1000.0

    suite.bench(
        "partition(arr, kth=50000) [100k Float64]",
        lambda: np.partition(rand_vec, 50000),
        iterations=200,
    )
    suite.bench(
        "argpartition(arr, kth=50000) [100k Float64]",
        lambda: np.argpartition(rand_vec, 50000),
        iterations=200,
    )

    sorted_target = np.linspace(0.0, 1000.0, size, dtype=np.float64)
    suite.bench(
        "searchsorted(sorted, queries) [100k in 100k]",
        lambda: np.searchsorted(sorted_target, rand_vec),
        iterations=150,
    )

    sparse_arr = np.array(
        [float(i + 1) if i % 10 == 0 else 0.0 for i in range(size)],
        dtype=np.float64,
    )

    suite.bench(
        "nonzero(sparse) [100k Float64, 10% nonzero]",
        lambda: np.nonzero(sparse_arr),
        iterations=300,
    )
    suite.bench(
        "argwhere(sparse) [100k Float64, 10% nonzero]",
        lambda: np.argwhere(sparse_arr),
        iterations=300,
    )
    suite.bench(
        "count_nonzero(sparse) [100k Float64]",
        lambda: np.count_nonzero(sparse_arr),
        iterations=500,
    )
    suite.bench(
        "argmax(arr) [100k Float64]",
        lambda: np.argmax(rand_vec),
        iterations=500,
    )
    suite.bench(
        "argmin(arr) [100k Float64]",
        lambda: np.argmin(rand_vec),
        iterations=500,
    )

    suite.finish()


if __name__ == "__main__":
    main()
