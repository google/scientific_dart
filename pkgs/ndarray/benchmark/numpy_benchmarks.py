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

import time
import numpy as np

def run_benchmark(name, setup_fn, run_fn, iterations=100):
    # Warmup
    setup_fn()
    for _ in range(10):
        run_fn()
        
    times = []
    for _ in range(iterations):
        setup_fn()
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1_000_000) # Convert to microseconds
    return np.mean(times)

print("============================================================================")
print("         NumPy EQUIVALENT PERFORMANCE BENCHMARK SUITE")
print("============================================================================")

# TRACK A: RNG
rng = np.random.default_rng(42)

# 1. Normal
def run_normal():
    rng.normal(loc=0.0, scale=1.0, size=50000)
t_normal = run_benchmark("Normal", lambda: None, run_normal, 200)
print(f"RNG Track  | Seeded normal() (Gaussian samples)       [size=50,000]: {t_normal:.2f} us")

# 2. Poisson
def run_poisson():
    rng.poisson(lam=35.0, size=20000)
t_poisson = run_benchmark("Poisson", lambda: None, run_poisson, 200)
print(f"RNG Track  | Seeded poisson() (Knuth vs Gaussian)      [size=20,000]: {t_poisson:.2f} us")

# 3. Binomial
def run_binomial():
    rng.binomial(n=60, p=0.4, size=20000)
t_binomial = run_benchmark("Binomial", lambda: None, run_binomial, 200)
print(f"RNG Track  | Seeded binomial() (Bernoulli vs Normal)   [size=20,000]: {t_binomial:.2f} us")

print("\n--- TRACK B: NATIVE C HEAP SORTING & SEARCHING ---")

# 1. Sort
sort_arr = np.zeros(30000, dtype=np.float64)
template_sort = np.arange(30000, 0, -1, dtype=np.float64)
def run_sort():
    np.sort(sort_arr)
t_sort = run_benchmark("Sort", lambda: np.copyto(sort_arr, template_sort), run_sort, 100)
print(f"SORT Track | NumPy sort() (Contiguous vector)         [size=30,000]: {t_sort:.2f} us")

# 1b. Sort Random
rng = np.random.default_rng(42)
template_sort_random = rng.random(30000)
sort_arr_random = np.zeros(30000, dtype=np.float64)
def run_sort_random():
    np.sort(sort_arr_random)
t_sort_random = run_benchmark("SortRandom", lambda: np.copyto(sort_arr_random, template_sort_random), run_sort_random, 100)
print(f"SORT Track | NumPy sort() (Random vector)             [size=30,000]: {t_sort_random:.2f} us")

# 1c. Boolean Mask Indexing
target_mask_arr = np.zeros(100000, dtype=np.float64)
rng_mask = np.random.default_rng(42)
mask_arr = rng_mask.random(100000) > 0.5
def run_mask():
    target_mask_arr[mask_arr]
t_mask = run_benchmark("BooleanMask", lambda: None, run_mask, 100)
print(f"INDEX Track | NumPy Boolean Mask Indexing              [size=100,000]: {t_mask:.2f} us")

# 2. Argsort
def run_argsort():
    np.argsort(sort_arr)
t_argsort = run_benchmark("Argsort", lambda: np.copyto(sort_arr, template_sort), run_argsort, 100)
print(f"SORT Track | Argsort (argsort)                         [size=30,000]: {t_argsort:.2f} us")

# 3. Where
cond_where = np.zeros((100, 100), dtype=bool)
x_where = np.ones(100, dtype=np.float64)
y_where = np.ones((100, 100), dtype=np.float64)
def run_where():
    np.where(cond_where, x_where, y_where)
t_where = run_benchmark("Where", lambda: None, run_where, 500)
print(f"SORT Track | Ternary where() 3-Way Broadcasting       [shape=100x100]: {t_where:.2f} us")

print("\n--- TRACK C: LINEAR ALGEBRA & SIGNALS ---")

# 1. Inversion
inv_arr = np.eye(100, dtype=np.float64)
def run_inv():
    np.linalg.inv(inv_arr)
t_inv = run_benchmark("Inv", lambda: None, run_inv, 200)
print(f"LINALG Track| LU Matrix Inversion (inv)                 [shape=100x100]: {t_inv:.2f} us")

# 2. QR
qr_arr = np.zeros((30, 30), dtype=np.float64)
for i in range(30):
    for j in range(30):
        qr_arr[i, j] = (i + j + 1.0) / 10.0
        if i == j:
            qr_arr[i, j] += 1.0
def run_qr():
    np.linalg.qr(qr_arr)
t_qr = run_benchmark("QR", lambda: None, run_qr, 500)
print(f"LINALG Track| QR Decomposition (qr)                    [shape=30x30]: {t_qr:.2f} us")

# 3. SVD
def run_svd():
    np.linalg.svd(qr_arr)
t_svd = run_benchmark("SVD", lambda: None, run_svd, 500)
print(f"LINALG Track| SVD Decomposition (svd)                  [shape=30x30]: {t_svd:.2f} us")

# 4. FFT
fft_arr = np.zeros(2048, dtype=np.float64)
def run_fft():
    np.fft.fft(fft_arr)
t_fft = run_benchmark("FFT", lambda: None, run_fft, 500)
print(f"LINALG Track| FFT pocketfft (fft)                       [length=2048]: {t_fft:.2f} us")

# 5. Cholesky
cholesky_arr = np.zeros((30, 30), dtype=np.float64)
for i in range(30):
    for j in range(30):
        cholesky_arr[i, j] = (i + j + 1.0) / 10.0
        if i == j:
            cholesky_arr[i, j] += 30.0
def run_cholesky():
    np.linalg.cholesky(cholesky_arr)
t_cholesky = run_benchmark("Cholesky", lambda: None, run_cholesky, 500)
print(f"LINALG Track| Cholesky Decomposition (cholesky)       [shape=30x30]: {t_cholesky:.2f} us")

# 6. Matmul
matmul_a = np.ones((100, 100), dtype=np.float64)
matmul_b = np.ones((100, 100), dtype=np.float64)
def run_matmul():
    np.matmul(matmul_a, matmul_b)
t_matmul = run_benchmark("Matmul", lambda: None, run_matmul, 500)
print(f"LINALG Track| Matrix Multiplication (matmul)          [shape=100x100]: {t_matmul:.2f} us")


print("\n--- TRACK D: UNIVERSAL UFUNCS, REDUCTIONS & MEMORY STRIDES ---")

# 1. Elementwise Add
add_x = np.ones(300000, dtype=np.float64)
add_y = np.ones(300000, dtype=np.float64)
add_out = np.zeros(300000, dtype=np.float64)
def run_add():
    np.add(add_x, add_y, out=add_out)
t_add = run_benchmark("Add", lambda: None, run_add, 500)
print(f"MEMORY Track| Element-wise Same-Shape add(x, y)       [size=300,000]: {t_add:.2f} us")

# 2. Scalar Broadcast Add
scalar_arr = np.array([5.0], dtype=np.float64)
def run_scalar_add():
    np.add(add_x, scalar_arr, out=add_out)
t_scalar_add = run_benchmark("ScalarAdd", lambda: None, run_scalar_add, 500)
print(f"MEMORY Track| Scalar Array Broadcast add(x, scalar)   [size=300,000]: {t_scalar_add:.2f} us")

# 3. Sin
sin_x = np.ones(100000, dtype=np.float64)
sin_out = np.zeros(100000, dtype=np.float64)
def run_sin():
    np.sin(sin_x, out=sin_out)
t_sin = run_benchmark("Sin", lambda: None, run_sin, 500)
print(f"MEMORY Track| Universal math function sin(x)          [size=100,000]: {t_sin:.2f} us")

# 4. Cos
def run_cos():
    np.cos(sin_x, out=sin_out)
t_cos = run_benchmark("Cos", lambda: None, run_cos, 500)
print(f"MEMORY Track| Universal math function cos(x)          [size=100,000]: {t_cos:.2f} us")

# 5. Exp
def run_exp():
    np.exp(sin_x, out=sin_out)
t_exp = run_benchmark("Exp", lambda: None, run_exp, 500)
print(f"MEMORY Track| Universal math function exp(x)          [size=100,000]: {t_exp:.2f} us")

# 6. Sum
def run_sum():
    np.sum(add_x)
t_sum = run_benchmark("Sum", lambda: None, run_sum, 500)
print(f"MEMORY Track| Flat Memory Reduction walk sum(x)       [size=300,000]: {t_sum:.2f} us")

# 7. Zeros
def run_zeros():
    np.zeros((1000, 1000), dtype=np.float64)
t_zeros = run_benchmark("Zeros", lambda: None, run_zeros, 200)
print(f"MEMORY Track| Zeros Array Creation (zeros)         [size=1,000,000]: {t_zeros:.2f} us")

# 8. Concatenate
cat_a = np.ones(500000, dtype=np.float64)
cat_b = np.ones(500000, dtype=np.float64)
def run_concat():
    np.concatenate((cat_a, cat_b), axis=0)
t_concat = run_benchmark("Concat", lambda: None, run_concat, 100)
print(f"MEMORY Track| Flat Array Concatenation (concatenate) [size=1,000,000]: {t_concat:.2f} us")

# 9. Clip
def run_clip():
    np.clip(add_x, 0.0, 0.5, out=add_out)
t_clip = run_benchmark("Clip", lambda: None, run_clip, 500)
print(f"MEMORY Track| Universal math function clip(x)          [size=300,000]: {t_clip:.2f} us")

# 10. Flatten
parent_flat = np.ones(600000, dtype=np.float64)
view_flat = parent_flat[0:300000]
def run_flatten():
    view_flat.flatten()
t_flatten = run_benchmark("Flatten", lambda: None, run_flatten, 500)
print(f"MEMORY Track| Contiguous View Flatten (flatten)       [size=300,000]: {t_flatten:.2f} us")

# 11. Contiguous view sum
def run_view_sum():
    np.sum(view_flat)
t_view_sum = run_benchmark("ViewSum", lambda: None, run_view_sum, 500)
print(f"MEMORY Track| Contiguous View Sum Reduction (sum)     [size=300,000]: {t_view_sum:.2f} us")

# 12. Strided transposed add
parent_strided_x = np.ones((500, 500), dtype=np.float64)
parent_strided_y = np.ones((500, 500), dtype=np.float64)
strided_x = parent_strided_x.T
strided_y = parent_strided_y.T
strided_out = np.zeros((500, 500), dtype=np.float64)
def run_strided_add():
    np.add(strided_x, strided_y, out=strided_out)
t_strided_add = run_benchmark("StridedAdd", lambda: None, run_strided_add, 200)
print(f"MEMORY Track| Strided non-contiguous add(x, y)         [shape=500x500]: {t_strided_add:.2f} us")

print("\n--- TRACK E: DISTANCE METRICS ---")

# Setup data
rng_dist = np.random.default_rng(42)
x_dist = rng_dist.normal(size=(500, 100))
# for hamming
x_dist_int = rng_dist.integers(0, 2, size=(500, 100), dtype=np.int32)

xa_dist = rng_dist.normal(size=(500, 100))
xb_dist = rng_dist.normal(size=(500, 100))

xa_dist_int = rng_dist.integers(0, 2, size=(500, 100), dtype=np.int32)
xb_dist_int = rng_dist.integers(0, 2, size=(500, 100), dtype=np.int32)

# pdist Euclidean
def run_pdist_euclidean():
    # We use broadcasting Euclidean
    M = x_dist.shape[0]
    dists = np.sqrt(np.sum((x_dist[:, np.newaxis, :] - x_dist[np.newaxis, :, :]) ** 2, axis=-1))
    _ = dists[np.triu_indices(M, k=1)]

t_pdist_euc = run_benchmark("PdistEuclidean", lambda: None, run_pdist_euclidean, 50)
print(f"DISTANCE Track| NumPy pdist Euclidean                          [shape=500x100]: {t_pdist_euc:.2f} us")

# pdist Cosine
def run_pdist_cosine():
    M = x_dist.shape[0]
    dot_product = np.dot(x_dist, x_dist.T)
    norm_x = np.linalg.norm(x_dist, axis=1)
    norm_x[norm_x == 0] = np.nan
    dists = 1.0 - dot_product / (norm_x[:, np.newaxis] * norm_x[np.newaxis, :])
    _ = dists[np.triu_indices(M, k=1)]

t_pdist_cos = run_benchmark("PdistCosine", lambda: None, run_pdist_cosine, 50)
print(f"DISTANCE Track| NumPy pdist Cosine                             [shape=500x100]: {t_pdist_cos:.2f} us")

# pdist Hamming
def run_pdist_hamming():
    M = x_dist_int.shape[0]
    dists = np.mean(x_dist_int[:, np.newaxis, :] != x_dist_int[np.newaxis, :, :], axis=-1)
    _ = dists[np.triu_indices(M, k=1)]

t_pdist_ham = run_benchmark("PdistHamming", lambda: None, run_pdist_hamming, 50)
print(f"DISTANCE Track| NumPy pdist Hamming                             [shape=500x100]: {t_pdist_ham:.2f} us")

# cdist Euclidean
def run_cdist_euclidean():
    _ = np.sqrt(np.sum((xa_dist[:, np.newaxis, :] - xb_dist[np.newaxis, :, :]) ** 2, axis=-1))

t_cdist_euc = run_benchmark("CdistEuclidean", lambda: None, run_cdist_euclidean, 50)
print(f"DISTANCE Track| NumPy cdist Euclidean                          [shape=500x100 vs 500x100]: {t_cdist_euc:.2f} us")

# cdist Cosine
def run_cdist_cosine():
    dot_product = np.dot(xa_dist, xb_dist.T)
    norm_a = np.linalg.norm(xa_dist, axis=1)
    norm_b = np.linalg.norm(xb_dist, axis=1)
    norm_a[norm_a == 0] = np.nan
    norm_b[norm_b == 0] = np.nan
    _ = 1.0 - dot_product / (norm_a[:, np.newaxis] * norm_b[np.newaxis, :])

t_cdist_cos = run_benchmark("CdistCosine", lambda: None, run_cdist_cosine, 50)
print(f"DISTANCE Track| NumPy cdist Cosine                             [shape=500x100 vs 500x100]: {t_cdist_cos:.2f} us")

# cdist Hamming
def run_cdist_hamming():
    _ = np.mean(xa_dist_int[:, np.newaxis, :] != xb_dist_int[np.newaxis, :, :], axis=-1)

t_cdist_ham = run_benchmark("CdistHamming", lambda: None, run_cdist_hamming, 50)
print(f"DISTANCE Track| NumPy cdist Hamming                             [shape=500x100 vs 500x100]: {t_cdist_ham:.2f} us")

