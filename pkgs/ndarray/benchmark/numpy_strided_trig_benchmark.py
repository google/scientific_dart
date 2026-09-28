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
        "NumPy Non-Contiguous sin() Performance Benchmark",
        "strided_trig",
    )

    mat = (np.arange(2000 * 2000, dtype=np.float64) / 10000.0).reshape(
        (2000, 2000)
    )
    mat_t = mat.T
    out = np.empty((2000, 2000), dtype=np.float64)

    suite.bench(
        "strided sin(matT) [shape=2000x2000 transposed]",
        lambda: np.sin(mat_t, out=out),
        iterations=25,
        warmup=4,
    )

    suite.finish()


if __name__ == "__main__":
    main()
