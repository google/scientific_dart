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

def benchmark(name, run_fn, iterations=200, warmup=20):
    for _ in range(warmup):
        run_fn()
    t0 = time.perf_counter()
    for _ in range(iterations):
        run_fn()
    t1 = time.perf_counter()
    us_per_op = ((t1 - t0) / iterations) * 1_000_000.0
    print(f"NumPy {name:<45}: {us_per_op:8.2f} us/op")
    return us_per_op

print("============================================================================")
print("         NumPy Benchmark Suite for Refactored & New Operations              ")
print("============================================================================")

# 1. einsum (100x100 matrix multiplication)
a_mat = np.arange(10000, dtype=np.float64).reshape((100, 100))
b_mat = np.arange(10000, dtype=np.float64).reshape((100, 100))
benchmark("einsum matrix mult ('ij,jk->ik') [100x100]", lambda: np.einsum('ij,jk->ik', a_mat, b_mat), iterations=500)

# 2. einsum ellipsis broadcasting (10x20x20)
a_3d = np.arange(4000, dtype=np.float64).reshape((10, 20, 20))
b_3d = np.arange(4000, dtype=np.float64).reshape((10, 20, 20))
benchmark("einsum batch matmul ('...ij,...jk->...ik') [10x20x20]", lambda: np.einsum('...ij,...jk->...ik', a_3d, b_3d), iterations=500)

c_mat = np.arange(10000, dtype=np.float64).reshape((100, 100))
benchmark("einsum 3-operand ('ij,jk,kl->il') [100x100]", lambda: np.einsum('ij,jk,kl->il', a_mat, b_mat, c_mat), iterations=100)


# 3. tensordot count=2 (100x100 matrix dot)
benchmark("tensordot axes=2 [100x100]", lambda: np.tensordot(a_mat, b_mat, axes=2), iterations=500)

# 4. tensordot explicit ([1],[0]) (100x100 matrix dot)
benchmark("tensordot axes=([1],[0]) [100x100]", lambda: np.tensordot(a_mat, b_mat, axes=([1],[0])), iterations=500)

# 5. correlate 1D full mode (length 10,000 array, length 100 kernel)
a_1d = np.arange(10000, dtype=np.float64)
v_1d = np.arange(100, dtype=np.float64)
benchmark("correlate mode='full' [N=10,000, K=100]", lambda: np.correlate(a_1d, v_1d, mode='full'), iterations=200)

# 6. convolve 1D full mode (length 10,000 array, length 100 kernel)
benchmark("convolve mode='full' [N=10,000, K=100]", lambda: np.convolve(a_1d, v_1d, mode='full'), iterations=200)
