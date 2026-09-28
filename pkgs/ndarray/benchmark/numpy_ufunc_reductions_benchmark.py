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

def main():
    size1d = 1000000
    dim2d = 1000
    iterations = 100
    warmup = 10

    print("================================================================")
    print("NumPy (Python) Ufunc Reductions & Masked Functions Benchmark")
    print("================================================================")
    print(f"Array Size 1D: {size1d} elements")
    print(f"Array Size 2D: {dim2d}x{dim2d} elements")
    print(f"Iterations: {iterations} (after {warmup} warmup runs)\n")

    a1d = np.array([i % 100 * 1.0 for i in range(size1d)], dtype=np.float64)
    b1d = np.array([(i % 50 + 1) * 1.0 for i in range(size1d)], dtype=np.float64)
    mask1d = np.array([1 if i % 2 == 0 else 0 for i in range(size1d)], dtype=bool)

    a2d = np.array([(i % 100) * 1.0 for i in range(dim2d * dim2d)], dtype=np.float64).reshape((dim2d, dim2d))
    b2d = np.array([(i % 50 + 1) * 1.0 for i in range(dim2d * dim2d)], dtype=np.float64).reshape((dim2d, dim2d))

    aOuter = np.array([i * 1.0 for i in range(1000)], dtype=np.float64)
    bOuter = np.array([(i + 1) * 1.0 for i in range(1000)], dtype=np.float64)

    indicesAt = np.array([(i * 97) % size1d for i in range(10000)], dtype=np.int64)
    valsAt = np.array([i * 1.0 for i in range(10000)], dtype=np.float64)

    indicesReduceat = np.array([i * 1000 for i in range(1000)], dtype=np.int64)

    out1d = np.zeros(size1d, dtype=np.float64)
    out2d = np.zeros((dim2d, dim2d), dtype=np.float64)

    def bench(name, fn):
        for _ in range(warmup):
            fn()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / iterations) * 1e6
        avg_ms = avg_us / 1000.0
        print(f"{name:<42}: {avg_us:>10.2f} us ({avg_ms:>6.3f} ms)")
        return avg_us

    # 1. Reductions
    bench("reduce(add) [1D 1M global]", lambda: np.add.reduce(a1d))
    bench("reduce(add) [2D 1000x1000 axis:0]", lambda: np.add.reduce(a2d, axis=0))
    bench("reduce(multiply) [1D 100K global]", lambda: np.multiply.reduce(a1d[:100000]))

    # 2. Accumulate
    bench("accumulate(add) [1D 1M cumsum]", lambda: np.add.accumulate(a1d))
    bench("accumulate(add) [2D 1000x1000 axis:0]", lambda: np.add.accumulate(a2d, axis=0))

    # 3. Outer
    bench("outer(add) [1000 x 1000]", lambda: np.add.outer(aOuter, bOuter))
    bench("outer(multiply) [1000 x 1000]", lambda: np.multiply.outer(aOuter, bOuter))

    # 4. Reduceat
    bench("reduceat(add) [1M array, 1000 segments]", lambda: np.add.reduceat(a1d, indicesReduceat))

    # 5. At (unbuffered scatter update)
    bench("at(add) [1M array, 10K indices]", lambda: np.add.at(a1d, indicesAt, valsAt))

    # 6. Masked Functions (where=)
    bench("add(where=mask) [1D 1M contiguous]", lambda: np.add(a1d, b1d, where=mask1d, out=out1d))
    bench("multiply(where=mask) [1D 1M contiguous]", lambda: np.multiply(a1d, b1d, where=mask1d, out=out1d))

    aStrided = a2d.ravel()
    bStrided = b2d.ravel()
    bench("add(where=mask) [1D 1M strided view]", lambda: np.add(aStrided[::2], bStrided[::2]))

if __name__ == "__main__":
    main()
