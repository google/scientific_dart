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
        "NumPy Real FFT, 2D Transform & Window Functions Benchmark Suite",
        "fft_dsp",
    )

    suite.group("1. Real-Valued 1D Transforms (rfft & irfft)")
    for length in [1024, 4096, 16384, 65536]:
        real_signal = np.linspace(0.0, 100.0, length, dtype=np.float64)
        spec_input = np.fft.rfft(real_signal)

        suite.bench(
            f"rfft(realSignal) [length={length}]",
            lambda sig=real_signal: np.fft.rfft(sig),
            iterations=200 if length <= 16384 else 100,
        )
        suite.bench(
            f"irfft(spec) [length={length}]",
            lambda sp=spec_input, n=length: np.fft.irfft(sp, n=n),
            iterations=200 if length <= 16384 else 100,
        )

    suite.group("2. 2D Complex Fourier Transforms (fft2 & ifft2)")
    for dim in [256, 512]:
        img2d = np.eye(dim, dtype=np.float64)
        img_spec = np.fft.fft2(img2d)

        suite.bench(
            f"fft2 [{dim}x{dim}]",
            lambda im=img2d: np.fft.fft2(im),
            iterations=100 if dim == 256 else 40,
        )
        suite.bench(
            f"ifft2 [{dim}x{dim}]",
            lambda sp=img_spec: np.fft.ifft2(sp),
            iterations=100 if dim == 256 else 40,
        )

    suite.group("3. DSP Window Functions")
    window_size = 100000
    suite.bench(
        f"hanning({window_size})",
        lambda: np.hanning(window_size),
        iterations=200,
    )
    suite.bench(
        f"hamming({window_size})",
        lambda: np.hamming(window_size),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()
