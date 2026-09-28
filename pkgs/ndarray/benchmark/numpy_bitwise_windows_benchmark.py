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

    suite = NumpyBenchSuite(
        "NumPy Bitwise, Windows & Special Functions Benchmark Suite",
        "bitwise_windows",
    )

    suite.group("1. DSP Windowing Functions (100k points)")
    suite.bench("hanning(100k)", lambda: np.hanning(size), iterations=300)
    suite.bench("hamming(100k)", lambda: np.hamming(size), iterations=300)

    suite.group("2. Special Mathematical Functions (100k elements)")
    float_vec = np.linspace(0.0, 10.0, size, dtype=np.float64)
    suite.bench(
        "i0(x) (Bessel I0) [100k]", lambda: np.i0(float_vec), iterations=200
    )
    suite.bench(
        "sinc(x) (Normalized Sinc) [100k]",
        lambda: np.sinc(float_vec),
        iterations=200,
    )

    suite.group("3. Bitwise Integer Operations (100k elements)")
    int_a = (np.arange(size, dtype=np.int32) * 13).astype(np.int32)
    int_b = (np.arange(size, dtype=np.int32) * 7 + 1).astype(np.int32)

    suite.bench(
        "bitwise_and(a, b) [100k Int32]",
        lambda: np.bitwise_and(int_a, int_b),
        iterations=500,
    )
    suite.bench(
        "bitwise_or(a, b) [100k Int32]",
        lambda: np.bitwise_or(int_a, int_b),
        iterations=500,
    )
    suite.bench(
        "bitwise_xor(a, b) [100k Int32]",
        lambda: np.bitwise_xor(int_a, int_b),
        iterations=500,
    )
    suite.bench(
        "invert(a) [100k Int32]",
        lambda: np.invert(int_a),
        iterations=500,
    )

    shift_amt = (np.arange(size, dtype=np.int32) % 8).astype(np.int32)
    suite.bench(
        "left_shift(a, shift) [100k Int32]",
        lambda: np.left_shift(int_a, shift_amt),
        iterations=500,
    )
    suite.bench(
        "right_shift(a, shift) [100k Int32]",
        lambda: np.right_shift(int_a, shift_amt),
        iterations=500,
    )

    suite.finish()


if __name__ == "__main__":
    main()
