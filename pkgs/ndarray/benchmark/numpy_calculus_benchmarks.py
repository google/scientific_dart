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
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000) # Convert to milliseconds
    
    avg = np.mean(times)
    # BenchmarkBase.report() reports in microseconds per call, 
    # but the output string says "μs".
    # Actually benchmark_harness reports "runs per second" usually if using emitter, 
    # but .report() prints: "Name(RunTime): XXX.XXX us."
    return avg * 1000 # returns microseconds

print("============================================================================")
print("         NumPy CALCULUS PERFORMANCE BENCHMARK SUITE")
print("============================================================================")

# 1. Trapz 1D (1M)
y_1d = np.arange(1000000, dtype=np.float64)
def run_trapz_1d():
    np.trapz(y_1d)
t_trapz = run_benchmark("Trapz 1D", lambda: None, run_trapz_1d, 100)
print(f"Calculus | trapz 1D (Float64) [size=1,000,000](RunTime): {t_trapz:.3f} us.")

# 2. Gradient 1D (1M)
f_1d = np.arange(1000000, dtype=np.float64) ** 2
def run_gradient_1d():
    np.gradient(f_1d)
t_grad_1d = run_benchmark("Gradient 1D", lambda: None, run_gradient_1d, 100)
print(f"Calculus | gradient 1D (Float64) [size=1,000,000](RunTime): {t_grad_1d:.3f} us.")

# 3. Gradient 2D (1000x1000)
f_2d = np.arange(1000000, dtype=np.float64).reshape(1000, 1000)
def run_gradient_2d():
    np.gradient(f_2d, axis=0)
t_grad_2d = run_benchmark("Gradient 2D", lambda: None, run_gradient_2d, 100)
print(f"Calculus | gradient 2D (Float64) [size=1,000x1,000](RunTime): {t_grad_2d:.3f} us.")
