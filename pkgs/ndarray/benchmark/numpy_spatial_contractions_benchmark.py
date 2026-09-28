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


def cdist_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    b_sq = np.sum(b * b, axis=1, keepdims=True).T
    sq_dist = np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(sq_dist)


def cdist_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True).T
    return 1.0 - (a @ b.T) / (norm_a * norm_b)


def pdist_euclidean(a: np.ndarray) -> np.ndarray:
    d = cdist_euclidean(a, a)
    return d[np.triu_indices(a.shape[0], k=1)]


def main():
    num_points = 500
    point_dim = 20

    suite = NumpyBenchSuite(
        "NumPy Spatial Metrics, Distance & Tensor Contractions Benchmark Suite",
        "spatial_contractions",
    )

    points_a = np.linspace(
        0.0, 10.0, num_points * point_dim, dtype=np.float64
    ).reshape((num_points, point_dim))
    points_b = np.linspace(
        5.0, 15.0, num_points * point_dim, dtype=np.float64
    ).reshape((num_points, point_dim))

    suite.group("1. Pairwise Spatial Distance Metrics")
    suite.bench(
        "cdist(A, B, euclidean) [500x20 vs 500x20 -> 500x500]",
        lambda: cdist_euclidean(points_a, points_b),
        iterations=100,
    )
    suite.bench(
        "cdist(A, B, cosine) [500x20 vs 500x20 -> 500x500]",
        lambda: cdist_cosine(points_a, points_b),
        iterations=100,
    )
    suite.bench(
        "pdist(A, euclidean) [500x20 -> 124,750 pairs]",
        lambda: pdist_euclidean(points_a),
        iterations=100,
    )

    suite.group("2. Tensor Dot & Contractions")
    mat_dim = 200
    mat_a = np.linspace(0.0, 10.0, mat_dim * mat_dim, dtype=np.float64).reshape(
        (mat_dim, mat_dim)
    )
    mat_b = np.linspace(5.0, 15.0, mat_dim * mat_dim, dtype=np.float64).reshape(
        (mat_dim, mat_dim)
    )

    suite.bench(
        'einsum("ij,jk->ik", [A, B]) [200x200 @ 200x200]',
        lambda: np.einsum("ij,jk->ik", mat_a, mat_b),
        iterations=100,
    )

    tensor_dim = 40
    t_a = np.linspace(
        0.0, 1.0, tensor_dim * tensor_dim * tensor_dim, dtype=np.float64
    ).reshape((tensor_dim, tensor_dim, tensor_dim))
    t_b = np.linspace(
        0.0, 1.0, tensor_dim * tensor_dim * tensor_dim, dtype=np.float64
    ).reshape((tensor_dim, tensor_dim, tensor_dim))

    suite.bench(
        "tensordot(tA, tB, axes=1) [40x40x40 @ 40x40x40]",
        lambda: np.tensordot(t_a, t_b, axes=1),
        iterations=100,
    )

    v_len = 1000
    v_a = np.linspace(0.0, 10.0, v_len, dtype=np.float64)
    v_b = np.linspace(5.0, 15.0, v_len, dtype=np.float64)
    suite.bench(
        "outer(vA, vB) [1000 x 1000 -> 1M]",
        lambda: np.outer(v_a, v_b),
        iterations=200,
    )

    kr_a = 50
    kr_b = 10
    k_a = np.linspace(0.0, 1.0, kr_a * kr_a, dtype=np.float64).reshape(
        (kr_a, kr_a)
    )
    k_b = np.linspace(0.0, 1.0, kr_b * kr_b, dtype=np.float64).reshape(
        (kr_b, kr_b)
    )
    suite.bench(
        "kron(kA, kB) [50x50 x 10x10 -> 500x500]",
        lambda: np.kron(k_a, k_b),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()
