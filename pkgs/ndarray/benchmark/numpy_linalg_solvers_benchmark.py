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


def make_invertible(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    a = (rng.random((n, n), dtype=np.float64) - 0.5) * 2.0
    a += np.eye(n, dtype=np.float64) * float(n)
    return a


def main():
    suite = NumpyBenchSuite(
        "NumPy Linear Algebra Solvers & Invariants Benchmark Suite",
        "linalg_solvers",
    )

    suite.group("1. Linear System Solvers (LAPACK dgesv & dgels)")
    for n in [50, 100, 200]:
        a_mat = make_invertible(n)
        b_vec = np.ones((n, 1), dtype=np.float64)
        suite.bench(
            f"solve(A, b) [{n}x{n}]",
            lambda a=a_mat, b=b_vec: np.linalg.solve(a, b),
            iterations=200 if n <= 100 else 100,
        )

    m = 200
    k = 50
    rng = np.random.default_rng(42)
    a_rect = rng.random((m, k), dtype=np.float64)
    b_rect = np.ones((m, 1), dtype=np.float64)
    suite.bench(
        "lstsq(A, b) [200x50]",
        lambda: np.linalg.lstsq(a_rect, b_rect, rcond=None),
        iterations=200,
    )

    suite.group("2. Determinants, Invariants & Matrix Norms")
    mat100 = make_invertible(100)

    suite.bench(
        "det(A) [100x100]",
        lambda: np.linalg.det(mat100),
        iterations=200,
    )
    suite.bench(
        "slogdet(A) [100x100]",
        lambda: np.linalg.slogdet(mat100),
        iterations=200,
    )
    suite.bench(
        "pinv(A) (Moore-Penrose SVD) [100x100]",
        lambda: np.linalg.pinv(mat100),
        iterations=100,
    )
    suite.bench(
        "norm(A) (Frobenius matrix norm) [100x100]",
        lambda: np.linalg.norm(mat100),
        iterations=500,
    )

    mat50 = make_invertible(50)
    suite.bench(
        "matrix_power(A, 5) [50x50]",
        lambda: np.linalg.matrix_power(mat50, 5),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()
