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
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    suite = NumpyBenchSuite(
        "NumPy Calculus Benchmarks",
        "calculus",
    )

    y_1d = np.arange(1000000, dtype=np.float64)
    f_1d = np.arange(1000000, dtype=np.float64) ** 2
    f_2d = np.arange(1000000, dtype=np.float64).reshape((1000, 1000))

    suite.bench(
        "Calculus | trapz 1D (Float64) [size=1,000,000]",
        lambda: trapz_fn(y_1d),
        iterations=100,
    )
    suite.bench(
        "Calculus | gradient 1D (Float64) [size=1,000,000]",
        lambda: np.gradient(f_1d),
        iterations=100,
    )
    suite.bench(
        "Calculus | gradient 2D (Float64) [size=1,000x1,000]",
        lambda: np.gradient(f_2d, axis=0),
        iterations=100,
    )

    suite.finish()


if __name__ == "__main__":
    main()
