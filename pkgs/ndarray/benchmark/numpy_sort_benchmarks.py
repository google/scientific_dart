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
        "NumPy Timsort & Argsort Comprehensive Benchmark Suite",
        "sort",
    )
    sizes = [1000, 10000, 50000]

    def register_track(label: str, gen_fn):
        suite.group(label)
        for size in sizes:
            template = gen_fn(size)
            target = np.empty_like(template)
            iters = 300 if size <= 10000 else 100

            suite.bench(
                f"Direct sort() [{size}]",
                lambda t=target: np.sort(t, kind="quicksort"),
                setup_fn=lambda dst=target, src=template: np.copyto(dst, src),
                iterations=iters,
            )
            suite.bench(
                f"Indirect argsort() [{size}]",
                lambda t=target: np.argsort(t, kind="quicksort"),
                setup_fn=lambda dst=target, src=template: np.copyto(dst, src),
                iterations=iters,
            )

    register_track(
        "Random Array",
        lambda sz: np.random.default_rng(42).random(sz, dtype=np.float64) * 1000.0,
    )
    register_track(
        "Already Sorted",
        lambda sz: np.arange(sz, dtype=np.float64),
    )
    register_track(
        "Reverse Sorted",
        lambda sz: np.arange(sz, 0, -1, dtype=np.float64),
    )

    suite.finish()


if __name__ == "__main__":
    main()
