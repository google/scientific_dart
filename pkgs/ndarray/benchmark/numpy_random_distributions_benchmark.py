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
        "NumPy Random Number Generation & Distributions Benchmark Suite",
        "random_distributions",
    )

    suite.group("1. Continuous & Discrete Distributions (100k samples)")
    suite.bench(
        "uniform([100k])",
        lambda: rng.uniform(0.0, 1.0, size=size),
        iterations=200,
    )
    suite.bench(
        "normal([100k], loc=5.0, scale=2.0)",
        lambda: rng.normal(loc=5.0, scale=2.0, size=size),
        iterations=200,
    )
    suite.bench(
        "exponential([100k], scale=1.5)",
        lambda: rng.exponential(scale=1.5, size=size),
        iterations=200,
    )
    suite.bench(
        "poisson([100k], lam=5.0)",
        lambda: rng.poisson(lam=5.0, size=size),
        iterations=150,
    )
    suite.bench(
        "binomial([100k], n=10, p=0.5)",
        lambda: rng.binomial(n=10, p=0.5, size=size),
        iterations=150,
    )
    suite.bench(
        "randint([100k], low=0, high=100)",
        lambda: rng.integers(0, 100, size=size, dtype=np.int64),
        iterations=200,
    )

    suite.group("2. Permutations, Choice & Shuffling")
    sample_pool = np.linspace(0.0, 100.0, size, dtype=np.float64)

    suite.bench(
        "choice(pool, size=100k, replace=true)",
        lambda: rng.choice(sample_pool, size=size, replace=True),
        iterations=200,
    )
    suite.bench(
        "permutation(arr) [100k]",
        lambda: rng.permutation(sample_pool),
        iterations=200,
    )

    shuffle_arr = np.linspace(0.0, 100.0, size, dtype=np.float64)
    suite.bench(
        "shuffle(arr) [100k in-place]",
        lambda: rng.shuffle(shuffle_arr),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()
