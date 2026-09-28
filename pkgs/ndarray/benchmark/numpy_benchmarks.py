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
        "NumPy ALL-INCLUSIVE PERFORMANCE BENCHMARK SUITE MASTER",
        "master",
    )

    rng = np.random.default_rng(42)

    suite.group("Track A: Random Distributions & RNG Solvers")
    suite.bench(
        "Seeded normal() (Gaussian samples) [size=50,000]",
        lambda: rng.normal(loc=0.0, scale=1.0, size=50000),
        iterations=200,
    )
    suite.bench(
        "Seeded poisson() (Knuth vs Gaussian) [size=20,000]",
        lambda: rng.poisson(lam=35.0, size=20000),
        iterations=200,
    )
    suite.bench(
        "Seeded binomial() (Bernoulli vs Normal) [size=20,000]",
        lambda: rng.binomial(n=60, p=0.4, size=20000),
        iterations=200,
    )

    suite.group("Track B: Native C Heap Sorting & Searching Broadcasts")
    sort_arr = np.zeros(30000, dtype=np.float64)
    template_sort = np.arange(30000, 0, -1, dtype=np.float64)
    suite.bench(
        "Native C Heap sort() (Contiguous vector) [size=30,000]",
        lambda: np.sort(sort_arr),
        setup_fn=lambda: np.copyto(sort_arr, template_sort),
        iterations=100,
    )

    template_sort_random = rng.random(30000)
    sort_arr_random = np.zeros(30000, dtype=np.float64)
    suite.bench(
        "Native C Heap sort() (Random vector) [size=30,000]",
        lambda: np.sort(sort_arr_random),
        setup_fn=lambda: np.copyto(sort_arr_random, template_sort_random),
        iterations=100,
    )

    target_mask_arr = np.zeros(100000, dtype=np.float64)
    mask_arr = rng.random(100000) > 0.5
    suite.bench(
        "Boolean Mask Advanced Indexing [size=100,000]",
        lambda: target_mask_arr[mask_arr],
        iterations=100,
    )

    suite.bench(
        "Argsort (argsort) [size=30,000]",
        lambda: np.argsort(sort_arr),
        setup_fn=lambda: np.copyto(sort_arr, template_sort),
        iterations=100,
    )

    cond_where = np.zeros((100, 100), dtype=bool)
    x_where = np.ones(100, dtype=np.float64)
    y_where = np.ones((100, 100), dtype=np.float64)
    suite.bench(
        "Ternary where() 3-Way Broadcasting [shape=100x100]",
        lambda: np.where(cond_where, x_where, y_where),
        iterations=500,
    )

    suite.group("Track C: OpenBLAS Linear Algebra & Native pocketfft Signals")
    inv_arr = np.eye(100, dtype=np.float64)
    suite.bench(
        "OpenBLAS LU Matrix Inversion (inv) [shape=100x100]",
        lambda: np.linalg.inv(inv_arr),
        iterations=200,
    )

    qr_arr = np.zeros((30, 30), dtype=np.float64)
    for i in range(30):
        for j in range(30):
            qr_arr[i, j] = (i + j + 1.0) / 10.0
            if i == j:
                qr_arr[i, j] += 1.0

    suite.bench(
        "QR Decomposition (qr) [shape=30x30]",
        lambda: np.linalg.qr(qr_arr),
        iterations=500,
    )
    suite.bench(
        "SVD Decomposition (svd) [shape=30x30]",
        lambda: np.linalg.svd(qr_arr),
        iterations=500,
    )

    fft_arr = np.zeros(2048, dtype=np.float64)
    suite.bench(
        "Native Mixed-Radix C FFI pocketfft (fft) [length=2048]",
        lambda: np.fft.fft(fft_arr),
        iterations=500,
    )

    cholesky_arr = np.zeros((30, 30), dtype=np.float64)
    for i in range(30):
        for j in range(30):
            cholesky_arr[i, j] = (i + j + 1.0) / 10.0
            if i == j:
                cholesky_arr[i, j] += 30.0

    suite.bench(
        "Cholesky Decomposition (cholesky) [shape=30x30]",
        lambda: np.linalg.cholesky(cholesky_arr),
        iterations=500,
    )

    matmul_a = np.ones((100, 100), dtype=np.float64)
    matmul_b = np.ones((100, 100), dtype=np.float64)
    suite.bench(
        "Matrix Multiplication (matmul) [shape=100x100]",
        lambda: np.matmul(matmul_a, matmul_b),
        iterations=500,
    )

    suite.group("Track D: Universal Ufuncs, Reductions & Memory Strides")
    add_x = np.ones(300000, dtype=np.float64)
    add_y = np.ones(300000, dtype=np.float64)
    add_out = np.zeros(300000, dtype=np.float64)
    suite.bench(
        "Element-wise Same-Shape add(x, y) [size=300,000]",
        lambda: np.add(add_x, add_y, out=add_out),
        iterations=500,
    )

    scalar_arr = np.array([5.0], dtype=np.float64)
    suite.bench(
        "Scalar Array Broadcast add(x, scalar) [size=300,000]",
        lambda: np.add(add_x, scalar_arr, out=add_out),
        iterations=500,
    )

    sin_x = np.ones(100000, dtype=np.float64)
    sin_out = np.zeros(100000, dtype=np.float64)
    suite.bench(
        "Universal math function sin(x) [size=100,000]",
        lambda: np.sin(sin_x, out=sin_out),
        iterations=500,
    )
    suite.bench(
        "Universal math function cos(x) [size=100,000]",
        lambda: np.cos(sin_x, out=sin_out),
        iterations=500,
    )
    suite.bench(
        "Universal math function exp(x) [size=100,000]",
        lambda: np.exp(sin_x, out=sin_out),
        iterations=500,
    )
    suite.bench(
        "Universal math function clip(x) [size=300,000]",
        lambda: np.clip(add_x, 0.0, 0.5, out=add_out),
        iterations=500,
    )
    suite.bench(
        "Flat Memory Reduction walk sum(x) [size=300,000]",
        lambda: np.sum(add_x),
        iterations=500,
    )
    suite.bench(
        "Zeros Array Creation (zeros) [size=1,000,000]",
        lambda: np.zeros((1000, 1000), dtype=np.float64),
        iterations=200,
    )

    cat_a = np.ones(500000, dtype=np.float64)
    cat_b = np.ones(500000, dtype=np.float64)
    suite.bench(
        "Flat Array Concatenation (concatenate) [size=1,000,000]",
        lambda: np.concatenate((cat_a, cat_b), axis=0),
        iterations=100,
    )

    parent_flat = np.ones(600000, dtype=np.float64)
    view_flat = parent_flat[0:300000]
    suite.bench(
        "Contiguous View Flatten (flatten) [size=300,000]",
        lambda: view_flat.flatten(),
        iterations=500,
    )
    suite.bench(
        "Contiguous View Sum Reduction (sum) [size=300,000]",
        lambda: np.sum(view_flat),
        iterations=500,
    )

    parent_strided_x = np.ones((500, 500), dtype=np.float64)
    parent_strided_y = np.ones((500, 500), dtype=np.float64)
    strided_x = parent_strided_x.T
    strided_y = parent_strided_y.T
    strided_out = np.zeros((500, 500), dtype=np.float64)
    suite.bench(
        "Strided non-contiguous add(x, y) [shape=500x500]",
        lambda: np.add(strided_x, strided_y, out=strided_out),
        iterations=200,
    )

    suite.group("Track E: Distance Metrics (pdist & cdist)")
    x_dist = rng.normal(size=(500, 100))
    x_dist_int = rng.integers(0, 2, size=(500, 100), dtype=np.int32)
    xa_dist = rng.normal(size=(500, 100))
    xb_dist = rng.normal(size=(500, 100))
    xa_dist_int = rng.integers(0, 2, size=(500, 100), dtype=np.int32)
    xb_dist_int = rng.integers(0, 2, size=(500, 100), dtype=np.int32)

    def run_pdist_euclidean():
        m = x_dist.shape[0]
        dists = np.sqrt(np.sum((x_dist[:, np.newaxis, :] - x_dist[np.newaxis, :, :]) ** 2, axis=-1))
        _ = dists[np.triu_indices(m, k=1)]

    def run_pdist_cosine():
        m = x_dist.shape[0]
        dot_product = np.dot(x_dist, x_dist.T)
        norm_x = np.linalg.norm(x_dist, axis=1)
        norm_x[norm_x == 0] = np.nan
        dists = 1.0 - dot_product / (norm_x[:, np.newaxis] * norm_x[np.newaxis, :])
        _ = dists[np.triu_indices(m, k=1)]

    def run_pdist_hamming():
        m = x_dist_int.shape[0]
        dists = np.mean(x_dist_int[:, np.newaxis, :] != x_dist_int[np.newaxis, :, :], axis=-1)
        _ = dists[np.triu_indices(m, k=1)]

    def run_cdist_euclidean():
        _ = np.sqrt(np.sum((xa_dist[:, np.newaxis, :] - xb_dist[np.newaxis, :, :]) ** 2, axis=-1))

    def run_cdist_cosine():
        dot_product = np.dot(xa_dist, xb_dist.T)
        norm_a = np.linalg.norm(xa_dist, axis=1)
        norm_b = np.linalg.norm(xb_dist, axis=1)
        norm_a[norm_a == 0] = np.nan
        norm_b[norm_b == 0] = np.nan
        _ = 1.0 - dot_product / (norm_a[:, np.newaxis] * norm_b[np.newaxis, :])

    def run_cdist_hamming():
        _ = np.mean(xa_dist_int[:, np.newaxis, :] != xb_dist_int[np.newaxis, :, :], axis=-1)

    suite.bench("pdist Euclidean [shape=500x100]", run_pdist_euclidean, iterations=50)
    suite.bench("pdist Cosine [shape=500x100]", run_pdist_cosine, iterations=50)
    suite.bench("pdist Hamming [shape=500x100 int32]", run_pdist_hamming, iterations=50)
    suite.bench("cdist Euclidean [shape=500x100 vs 500x100]", run_cdist_euclidean, iterations=50)
    suite.bench("cdist Cosine [shape=500x100 vs 500x100]", run_cdist_cosine, iterations=50)
    suite.bench("cdist Hamming [shape=500x100 vs 500x100 int32]", run_cdist_hamming, iterations=50)

    suite.finish()


if __name__ == "__main__":
    main()
