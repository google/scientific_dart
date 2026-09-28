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

import os
# Force single-threaded BLAS/LAPACK in NumPy to match Dart's setNumThreads(1)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import pathlib
import sys
import time
import numpy as np


class NumpyBenchSuite:
    def __init__(self, title: str, suite_id: str):
        self.title = title
        self.suite_id = suite_id
        self.current_group = ""
        self.results = []
        print("=" * 76)
        print(f" {self.title}")
        print("=" * 76)

    def group(self, name: str):
        self.current_group = name
        print(f"\n--- {name} ---")

    def bench(self, name: str, run_fn, setup_fn=None, iterations: int = 200, warmup: int = 15):
        if setup_fn is not None:
            setup_fn()
        for _ in range(warmup):
            if setup_fn is not None:
                setup_fn()
            run_fn()

        times_ns = []
        for _ in range(iterations):
            if setup_fn is not None:
                setup_fn()
            t0 = time.perf_counter_ns()
            run_fn()
            t1 = time.perf_counter_ns()
            times_ns.append(float(t1 - t0))

        arr_ns = np.array(times_ns, dtype=np.float64)
        mean_ns = float(np.mean(arr_ns))
        median_ns = float(np.median(arr_ns))
        std_ns = float(np.std(arr_ns))
        mean_us = mean_ns / 1000.0
        median_us = median_ns / 1000.0

        full_name = f"{self.current_group} / {name}" if self.current_group else name
        print(f"NumPy | {name:<54}: {mean_us:10.2f} us (median {median_us:10.2f} us)")

        entry = {
            "name": full_name,
            "short_name": name,
            "group": self.current_group,
            "iterations": iterations,
            "mean_ns": mean_ns,
            "median_ns": median_ns,
            "std_ns": std_ns,
            "mean_us": mean_us,
            "median_us": median_us,
        }
        self.results.append(entry)
        return mean_us

    def finish(self):
        script_dir = pathlib.Path(__file__).resolve().parent
        report_dir = script_dir / "report" / self.suite_id
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / "numpy_results.json"

        # Support optional --json <path> override
        if "--json" in sys.argv:
            idx = sys.argv.index("--json")
            if idx + 1 < len(sys.argv):
                out_path = pathlib.Path(sys.argv[idx + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "suite_id": self.suite_id,
                    "title": self.title,
                    "numpy_version": np.__version__,
                    "benchmarks": self.results,
                },
                f,
                indent=2,
            )
