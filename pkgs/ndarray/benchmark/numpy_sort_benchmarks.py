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

def run_sort_bench(label, size, template):
    iterations = 500 if size <= 10000 else 100
    
    # Warmup
    for _ in range(10):
        target = np.copy(template)
        np.sort(target, kind='stable')
        
    # Timed Direct Sort
    start = time.perf_counter()
    for _ in range(iterations):
        target = np.copy(template)
        np.sort(target, kind='stable')
    end = time.perf_counter()
    avg_sort_us = ((end - start) / iterations) * 1_000_000

    # Warmup argsort
    for _ in range(10):
        target = np.copy(template)
        np.argsort(target, kind='stable')

    # Timed Argsort
    start = time.perf_counter()
    for _ in range(iterations):
        target = np.copy(template)
        np.argsort(target, kind='stable')
    end = time.perf_counter()
    avg_argsort_us = ((end - start) / iterations) * 1_000_000

    print(f"{label:<16} | Direct sort(): {avg_sort_us:>10.2f} us | Argsort(): {avg_argsort_us:>10.2f} us")

if __name__ == "__main__":
    print("============================================================================")
    print("        PYTHON NUMPY TIMSORT & ARGSORT PERFORMANCE BENCHMARKS               ")
    print("============================================================================\n")

    sizes = [1000, 10000, 50000]
    for size in sizes:
        print("----------------------------------------------------------------------------")
        print(f" BENCHMARK METRICS FOR ARRAY SIZE: {size} elements")
        print("----------------------------------------------------------------------------")
        
        rng = np.random.default_rng(42)
        random_data = rng.random(size) * 1000.0
        sorted_data = np.arange(size, dtype=np.float64)
        reverse_data = np.arange(size, 0, -1, dtype=np.float64)
        
        run_sort_bench("Random Array", size, random_data)
        run_sort_bench("Already Sorted", size, sorted_data)
        run_sort_bench("Reverse Sorted", size, reverse_data)
        print("")
