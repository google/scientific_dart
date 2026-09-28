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
    suite = NumpyBenchSuite(
        "NumPy Benchmark Suite for Refactored & New Operations",
        "operations",
    )

    suite.group("Einsum Operations")
    a_mat = np.arange(10000, dtype=np.float64).reshape((100, 100))
    b_mat = np.arange(10000, dtype=np.float64).reshape((100, 100))
    out_mat = np.empty((100, 100), dtype=np.float64)

    suite.bench(
        "einsum matrix mult ('ij,jk->ik') [100x100]",
        lambda: np.einsum("ij,jk->ik", a_mat, b_mat),
        iterations=300,
    )
    suite.bench(
        "einsum matrix mult with out: [100x100]",
        lambda: np.einsum("ij,jk->ik", a_mat, b_mat, out=out_mat),
        iterations=300,
    )

    a_3d = np.arange(4000, dtype=np.float64).reshape((10, 20, 20))
    b_3d = np.arange(4000, dtype=np.float64).reshape((10, 20, 20))
    out_3d = np.empty((10, 20, 20), dtype=np.float64)

    suite.bench(
        "einsum batch matmul ('...ij,...jk->...ik') [10x20x20]",
        lambda: np.einsum("...ij,...jk->...ik", a_3d, b_3d),
        iterations=300,
    )
    suite.bench(
        "einsum batch matmul with out: [10x20x20]",
        lambda: np.einsum("...ij,...jk->...ik", a_3d, b_3d, out=out_3d),
        iterations=300,
    )

    c_mat = np.arange(10000, dtype=np.float64).reshape((100, 100))
    suite.bench(
        "einsum 3-operand ('ij,jk,kl->il') [100x100]",
        lambda: np.einsum("ij,jk,kl->il", a_mat, b_mat, c_mat),
        iterations=100,
    )

    suite.group("Tensordot Operations")
    suite.bench(
        "tensordot axes=2 [100x100]",
        lambda: np.tensordot(a_mat, b_mat, axes=2),
        iterations=500,
    )
    suite.bench(
        "tensordot axes=([1],[0]) [100x100]",
        lambda: np.tensordot(a_mat, b_mat, axes=([1], [0])),
        iterations=300,
    )

    suite.group("Convolution & Correlation")
    a_1d = np.arange(10000, dtype=np.float64)
    v_1d = np.arange(100, dtype=np.float64)
    suite.bench(
        "correlate mode='full' [N=10,000, K=100]",
        lambda: np.correlate(a_1d, v_1d, mode="full"),
        iterations=200,
    )
    suite.bench(
        "convolve mode='full' [N=10,000, K=100]",
        lambda: np.convolve(a_1d, v_1d, mode="full"),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()
