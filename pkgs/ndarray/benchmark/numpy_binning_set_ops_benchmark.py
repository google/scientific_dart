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
    size = 100000
    rng = np.random.default_rng(42)

    suite = NumpyBenchSuite(
        "NumPy Binning, Histograms & Set Operations Benchmark Suite",
        "binning_set_ops",
    )

    raw_data = rng.random(size, dtype=np.float64) * 100.0

    suite.group("1. Binning & Histograms")
    suite.bench(
        "histogram(data, bins: 100) [size=100,000]",
        lambda: np.histogram(raw_data, bins=100),
        iterations=200,
    )

    int_data = rng.integers(0, 500, size=size, dtype=np.int32)
    suite.bench(
        "bincount(intData) [size=100,000, 500 bins]",
        lambda: np.bincount(int_data),
        iterations=500,
    )

    bins = np.linspace(0.0, 100.0, 101, dtype=np.float64)
    suite.bench(
        "digitize(data, bins: 100) [size=100,000]",
        lambda: np.digitize(raw_data, bins),
        iterations=200,
    )

    suite.group("2. Set Operations")
    repeated_data = rng.integers(0, 10000, size=size, dtype=np.int32)
    suite.bench(
        "unique(data) [size=100,000, ~10,000 unique]",
        lambda: np.unique(repeated_data),
        iterations=200,
    )

    set_a = rng.integers(0, 50000, size=50000, dtype=np.int32)
    set_b = rng.integers(0, 50000, size=50000, dtype=np.int32)

    suite.bench(
        "isin(setA, setB) [50,000 vs 50,000]",
        lambda: np.isin(set_a, set_b),
        iterations=100,
    )
    suite.bench(
        "intersect1d(setA, setB) [50,000 vs 50,000]",
        lambda: np.intersect1d(set_a, set_b),
        iterations=100,
    )
    suite.bench(
        "union1d(setA, setB) [50,000 vs 50,000]",
        lambda: np.union1d(set_a, set_b),
        iterations=100,
    )

    suite.finish()


if __name__ == "__main__":
    main()
