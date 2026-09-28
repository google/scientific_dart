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

import shutil
import tempfile
from pathlib import Path
from numpy_bench_helper import NumpyBenchSuite, np


def main():
    temp_dir = Path(tempfile.mkdtemp(prefix="numpy_io_bench_"))
    npy_path = str(temp_dir / "array_1m.npy")
    npz_path = str(temp_dir / "archive.npz")
    npz_compressed_path = str(temp_dir / "archive_compressed.npz")

    element_count = 1000000

    try:
        suite = NumpyBenchSuite(
            "NumPy IO Serialization & Deserialization Benchmark Suite",
            "io",
        )

        raw_array = np.linspace(0.0, 100.0, element_count, dtype=np.float64)

        suite.group("1. Binary NumPy (.npy) File IO")
        suite.bench(
            "save() 8MB float64 array to .npy",
            lambda: np.save(npy_path, raw_array),
            iterations=40,
            warmup=5,
        )

        np.save(npy_path, raw_array)
        suite.bench(
            "load() 8MB float64 array from .npy",
            lambda: np.load(npy_path),
            iterations=80,
            warmup=5,
        )

        suite.group("2. NumPy Zip Archive (.npz) IO")
        half_array = np.linspace(
            0.0, 50.0, element_count // 2, dtype=np.float64
        )

        suite.bench(
            "savez() 12MB multi-array archive (.npz)",
            lambda: np.savez(npz_path, arr_a=raw_array, arr_b=half_array),
            iterations=30,
            warmup=4,
        )

        np.savez(npz_path, arr_a=raw_array, arr_b=half_array)

        def load_npz(path):
            with np.load(path) as data:
                _ = data["arr_a"]
                _ = data["arr_b"]

        suite.bench(
            "loadz() 12MB multi-array archive (.npz)",
            lambda: load_npz(npz_path),
            iterations=50,
            warmup=5,
        )

        suite.bench(
            "savez_compressed() Deflate 12MB (.npz)",
            lambda: np.savez_compressed(
                npz_compressed_path, arr_a=raw_array, arr_b=half_array
            ),
            iterations=15,
            warmup=3,
        )

        np.savez_compressed(
            npz_compressed_path, arr_a=raw_array, arr_b=half_array
        )
        suite.bench(
            "loadz() compressed Deflate 12MB (.npz)",
            lambda: load_npz(npz_compressed_path),
            iterations=25,
            warmup=4,
        )

        suite.finish()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
