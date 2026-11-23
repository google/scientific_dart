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

"""Curated NumPy / SciPy twin for `benchmark/readme_benchmark.dart`.

Uses the exact same deterministic Murmur3-based input generator, identical
warmup + batch-calibrated median/IQR timing methodology, and per-benchmark
verification checksums so results can be compared 1:1 with Dart AOT.
"""

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curated NumPy/SciPy benchmark twin for package:ndarray README."
    )
    parser.add_argument(
        "--mode",
        choices=("single", "multi"),
        default="single",
        help=(
            "Benchmark subset to run. Single- and multi-threaded modes must "
            "run in separate processes because OpenBLAS reads thread env vars "
            "once at import time."
        ),
    )
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--sample-ms", type=int, default=50)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--probe-blas",
        action="store_true",
        help="Print detected NumPy BLAS metadata as JSON and exit without running benchmarks.",
    )
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()
    if args.quick:
        args.samples = 5
        args.sample_ms = 15
        args.warmup_ms = 50
    return args


# Configure BLAS thread environment BEFORE importing numpy.
_ARGS = _parse_args()
if _ARGS.mode == "single":
    _thread_str = "1"
else:
    _thread_str = str(_ARGS.threads)

for _var in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_var] = _thread_str

import numpy as np  # noqa: E402

try:
    import scipy  # noqa: E402
    from scipy.spatial import distance as sp_distance  # noqa: E402

    _SCIPY_VERSION = scipy.__version__
except ImportError:
    scipy = None
    sp_distance = None
    _SCIPY_VERSION = None


def _detect_blas_info() -> dict:
    """Inspect NumPy's BLAS/LAPACK linkage and flag Netlib reference BLAS."""
    blas_name = "unknown"
    blas_lib = "unknown"
    is_reference_netlib = False

    try:
        cfg = np.__config__.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        blas_dep = deps.get("blas", {})
        if isinstance(blas_dep, dict):
            blas_name = str(blas_dep.get("name", "unknown"))
            openblas_cfg = blas_dep.get("openblas configuration")
            if openblas_cfg and str(openblas_cfg) != "unknown":
                blas_name = f"{blas_name} ({openblas_cfg})"
    except Exception:
        pass

    # On Linux, check actual dynamic linkage of `_umath_linalg` with `ldd`.
    if sys.platform.startswith("linux"):
        try:
            import numpy.linalg._umath_linalg as umath_linalg

            ldd_out = subprocess.check_output(
                ["ldd", umath_linalg.__file__],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in ldd_out.splitlines():
                stripped = line.strip()
                if "blas" in stripped.lower() or "mkl" in stripped.lower():
                    blas_lib = stripped
                    if "openblas" in stripped.lower():
                        if "openblas" not in blas_name.lower():
                            blas_name = "openblas"
                    elif "mkl" in stripped.lower():
                        blas_name = "mkl"
                    elif "blis" in stripped.lower():
                        blas_name = "blis"
                    elif "/libblas.so" in stripped:
                        resolved = stripped.split("=>")[-1].split("(")[0].strip()
                        real = os.path.realpath(resolved)
                        blas_lib = f"{resolved} -> {real}"
                        if (
                            "openblas" not in real.lower()
                            and "mkl" not in real.lower()
                            and "blis" not in real.lower()
                            and "atlas" not in real.lower()
                        ):
                            is_reference_netlib = True
                            blas_name = "netlib-reference-blas"
                    break
        except Exception:
            pass

    return {
        "blas_name": blas_name,
        "blas_lib": blas_lib,
        "is_reference_netlib": is_reference_netlib,
    }


def _mix32_array(count: int, seed: int) -> np.ndarray:
    """Bit-identical Murmur3 fmix32 over [seed, seed + count)."""
    with np.errstate(over="ignore"):
        h = np.arange(count, dtype=np.uint32) + np.uint32(seed)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0xC2B2AE35)
        h ^= h >> np.uint32(16)
    return h


def _uniform_f64(shape: tuple[int, ...], seed: int) -> np.ndarray:
    total = int(np.prod(shape))
    h = _mix32_array(total, seed)
    arr = (h.astype(np.float64) / 4294967296.0) * 2.0 - 1.0
    return arr.reshape(shape)


def _uniform_f32(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return _uniform_f64(shape, seed).astype(np.float32)


def _diag_dominant_f64(n: int, seed: int) -> np.ndarray:
    a = _uniform_f64((n, n), seed)
    a[np.arange(n), np.arange(n)] += float(n)
    return a


def _spd_f64(n: int, seed: int) -> np.ndarray:
    raw = _uniform_f64((n, n), seed)
    spd = 0.5 * (raw + raw.T)
    spd[np.arange(n), np.arange(n)] += float(n)
    return np.ascontiguousarray(spd, dtype=np.float64)


_SINK = 0.0


def _measure(
    *,
    bench_id: str,
    category: str,
    label: str,
    baseline: str,
    threads: int,
    check_tol: float,
    checksum_fn,
    bench_fn,
    args: argparse.Namespace,
) -> dict:
    global _SINK
    checksum = float(checksum_fn())
    if np.isfinite(checksum):
        _SINK += checksum

    warmup_target_ns = args.warmup_ms * 1_000_000
    t0 = time.perf_counter_ns()
    warm_calls = 0
    while warm_calls < 5 or (time.perf_counter_ns() - t0) < warmup_target_ns:
        bench_fn()
        warm_calls += 1
    warm_elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    est_call_us = max(warm_elapsed_us / warm_calls, 0.01)

    batch_size = max(1, min(100000, int(round((args.sample_ms * 1000.0) / est_call_us))))

    sample_us = []
    for _ in range(args.samples):
        s0 = time.perf_counter_ns()
        for _ in range(batch_size):
            bench_fn()
        s1 = time.perf_counter_ns()
        sample_us.append((s1 - s0) / 1000.0 / batch_size)

    arr = np.array(sample_us, dtype=np.float64)
    median_us = float(np.percentile(arr, 50.0))
    q1_us = float(np.percentile(arr, 25.0))
    q3_us = float(np.percentile(arr, 75.0))
    mean_us = float(np.mean(arr))

    tag = "SciPy" if baseline == "scipy" else "NumPy"
    print(
        f"{tag:<5s} | {label:<46s} | median {median_us:10.2f} us "
        f"(IQR {q3_us - q1_us:8.2f} us, batch={batch_size})"
    )

    return {
        "id": bench_id,
        "category": category,
        "label": label,
        "baseline": baseline,
        "threads": threads,
        "median_us": median_us,
        "iqr_us": q3_us - q1_us,
        "mean_us": mean_us,
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "batch_size": batch_size,
        "samples": args.samples,
        "checksum": checksum,
        "check_tol": check_tol,
    }


def _run_single(args: argparse.Namespace) -> list[dict]:
    results: list[dict] = []
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="numpy_readme_bench_"))

    try:
        # --- 1. Element-wise & Transcendentals ---
        a1m = _uniform_f64((1000000,), 101)
        b1m = _uniform_f64((1000000,), 202)
        out1m = np.empty((1000000,), dtype=np.float64)

        results.append(
            _measure(
                bench_id="add_inplace_1m",
                category="Element-wise & Transcendentals",
                label="add(a, b, out: out) [1M f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.sum(np.add(a1m, b1m, out=out1m)),
                bench_fn=lambda: np.add(a1m, b1m, out=out1m),
                args=args,
            )
        )

        results.append(
            _measure(
                bench_id="add_alloc_1m",
                category="Element-wise & Transcendentals",
                label="add(a, b) [1M f64, allocating]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.sum(np.add(a1m, b1m)),
                bench_fn=lambda: np.add(a1m, b1m),
                args=args,
            )
        )

        x100k = _uniform_f64((100000,), 303)
        out100k = np.empty((100000,), dtype=np.float64)

        results.append(
            _measure(
                bench_id="sin_100k",
                category="Element-wise & Transcendentals",
                label="sin(x, out: out) [100k f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-7,
                checksum_fn=lambda: np.sum(np.sin(x100k, out=out100k)),
                bench_fn=lambda: np.sin(x100k, out=out100k),
                args=args,
            )
        )

        results.append(
            _measure(
                bench_id="exp_100k",
                category="Element-wise & Transcendentals",
                label="exp(x, out: out) [100k f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-7,
                checksum_fn=lambda: np.sum(np.exp(x100k, out=out100k)),
                bench_fn=lambda: np.exp(x100k, out=out100k),
                args=args,
            )
        )

        mat_a1000 = _uniform_f64((1000, 1000), 404)
        mat_b1000 = _uniform_f64((1000, 1000), 505)
        mat_a_t = mat_a1000.T
        mat_b_t = mat_b1000.T
        out_mat1000 = np.empty((1000, 1000), dtype=np.float64)

        results.append(
            _measure(
                bench_id="strided_add_1000x1000",
                category="Element-wise & Transcendentals",
                label="add(a.T, b.T, out: out) [1000x1000 f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.sum(np.add(mat_a_t, mat_b_t, out=out_mat1000)),
                bench_fn=lambda: np.add(mat_a_t, mat_b_t, out=out_mat1000),
                args=args,
            )
        )

        # --- 2. Reductions & Statistics ---
        results.append(
            _measure(
                bench_id="sum_1m",
                category="Reductions & Statistics",
                label="sum(x) [1M f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.sum(a1m),
                bench_fn=lambda: np.sum(a1m),
                args=args,
            )
        )

        results.append(
            _measure(
                bench_id="mean_axis0_1000x1000",
                category="Reductions & Statistics",
                label="mean(mat, axis: 0) [1000x1000 f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.sum(np.mean(mat_a1000, axis=0)),
                bench_fn=lambda: np.mean(mat_a1000, axis=0),
                args=args,
            )
        )

        nan_vec = _uniform_f64((100000,), 606)
        nan_vec[::10] = np.nan

        results.append(
            _measure(
                bench_id="nansum_100k",
                category="Reductions & Statistics",
                label="nansum(x) [100k f64, 10% NaN]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.nansum(nan_vec),
                bench_fn=lambda: np.nansum(nan_vec),
                args=args,
            )
        )

        # --- 3. Linear Algebra & Tensor Contractions ---
        mm_a256 = _uniform_f64((256, 256), 707)
        mm_b256 = _uniform_f64((256, 256), 808)

        results.append(
            _measure(
                bench_id="matmul_256_f64",
                category="Linear Algebra & Tensor Contractions",
                label="matmul(A, B) [256x256 f64, 1T]",
                baseline="numpy",
                threads=1,
                check_tol=1e-7,
                checksum_fn=lambda: np.sum(np.matmul(mm_a256, mm_b256)),
                bench_fn=lambda: np.matmul(mm_a256, mm_b256),
                args=args,
            )
        )

        mm_a512 = _uniform_f32((512, 512), 909)
        mm_b512 = _uniform_f32((512, 512), 1010)

        results.append(
            _measure(
                bench_id="matmul_512_f32",
                category="Linear Algebra & Tensor Contractions",
                label="matmul(A, B) [512x512 f32, 1T]",
                baseline="numpy",
                threads=1,
                check_tol=1e-3,
                checksum_fn=lambda: np.sum(
                    np.matmul(mm_a512, mm_b512), dtype=np.float64
                ),
                bench_fn=lambda: np.matmul(mm_a512, mm_b512),
                args=args,
            )
        )

        solve_a = _diag_dominant_f64(200, 1111)
        solve_b = _uniform_f64((200, 1), 1212)

        results.append(
            _measure(
                bench_id="solve_200_f64",
                category="Linear Algebra & Tensor Contractions",
                label="solve(A, b) [200x200 f64, 1T]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.sum(np.linalg.solve(solve_a, solve_b)),
                bench_fn=lambda: np.linalg.solve(solve_a, solve_b),
                args=args,
            )
        )

        svd_a = _uniform_f64((128, 128), 1313)

        results.append(
            _measure(
                bench_id="svd_128_f64",
                category="Linear Algebra & Tensor Contractions",
                label="svd(A, fullMatrices: false) [128x128 f64, 1T]",
                baseline="numpy",
                threads=1,
                check_tol=1e-7,
                checksum_fn=lambda: np.sum(np.linalg.svd(svd_a, full_matrices=False)[1]),
                bench_fn=lambda: np.linalg.svd(svd_a, full_matrices=False),
                args=args,
            )
        )

        spd_a = _spd_f64(200, 1414)

        results.append(
            _measure(
                bench_id="cholesky_200_f64",
                category="Linear Algebra & Tensor Contractions",
                label="cholesky(A) [200x200 SPD f64, 1T]",
                baseline="numpy",
                threads=1,
                check_tol=1e-7,
                checksum_fn=lambda: np.sum(np.linalg.cholesky(spd_a)),
                bench_fn=lambda: np.linalg.cholesky(spd_a),
                args=args,
            )
        )

        attn_q = _uniform_f32((2, 8, 64, 64), 1515)
        attn_k = _uniform_f32((2, 8, 64, 64), 1616)

        results.append(
            _measure(
                bench_id="einsum_attn_f32",
                category="Linear Algebra & Tensor Contractions",
                label="einsum('bhid,bhjd->bhij') [2x8x64x64 f32, 1T]",
                baseline="numpy",
                threads=1,
                check_tol=1e-3,
                checksum_fn=lambda: np.sum(
                    np.einsum("bhid,bhjd->bhij", attn_q, attn_k, optimize=True),
                    dtype=np.float64,
                ),
                bench_fn=lambda: np.einsum(
                    "bhid,bhjd->bhij", attn_q, attn_k, optimize=True
                ),
                args=args,
            )
        )

        # --- 4. Spectral Transforms & Complex Math ---
        rfft_sig = _uniform_f64((65536,), 1717)

        results.append(
            _measure(
                bench_id="rfft_65536_f64",
                category="Spectral Transforms & Complex Math",
                label="rfft(x) [65,536 f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-5,
                checksum_fn=lambda: np.sum(np.abs(np.fft.rfft(rfft_sig))),
                bench_fn=lambda: np.fft.rfft(rfft_sig),
                args=args,
            )
        )

        img512 = _uniform_f64((512, 512), 1818)

        results.append(
            _measure(
                bench_id="fft2_512_f64",
                category="Spectral Transforms & Complex Math",
                label="fft2(img) [512x512 f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-4,
                checksum_fn=lambda: np.sum(np.abs(np.fft.fft2(img512))),
                bench_fn=lambda: np.fft.fft2(img512),
                args=args,
            )
        )

        c100k_raw = _uniform_f64((200000,), 1919)
        c100k = np.ascontiguousarray(c100k_raw.view(np.complex128))

        results.append(
            _measure(
                bench_id="complex_abs_100k",
                category="Spectral Transforms & Complex Math",
                label="abs(z) [100k complex128 -> f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-7,
                checksum_fn=lambda: np.sum(np.abs(c100k)),
                bench_fn=lambda: np.abs(c100k),
                args=args,
            )
        )

        # --- 5. Sorting & Selection ---
        sort_vec = _uniform_f64((100000,), 2020)

        def _sort_check():
            res = np.sort(sort_vec)
            return float(res[0] + res[50000] + res[99999])

        results.append(
            _measure(
                bench_id="sort_100k_f64",
                category="Sorting & Selection",
                label="sort(x) [100k random f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-12,
                checksum_fn=_sort_check,
                bench_fn=lambda: np.sort(sort_vec),
                args=args,
            )
        )

        def _argsort_check():
            res = np.argsort(sort_vec)
            return float(int(res[0]) + int(res[50000]) + int(res[99999]))

        results.append(
            _measure(
                bench_id="argsort_100k_f64",
                category="Sorting & Selection",
                label="argsort(x) [100k random f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-12,
                checksum_fn=_argsort_check,
                bench_fn=lambda: np.argsort(sort_vec),
                args=args,
            )
        )

        results.append(
            _measure(
                bench_id="partition_100k_f64",
                category="Sorting & Selection",
                label="partition(x, 50000) [100k random f64]",
                baseline="numpy",
                threads=1,
                check_tol=1e-12,
                checksum_fn=lambda: float(np.partition(sort_vec, 50000)[50000]),
                bench_fn=lambda: np.partition(sort_vec, 50000),
                args=args,
            )
        )

        # --- 6. Binary I/O (.npy) ---
        npy_path = str(temp_dir / "array_1m.npy")
        np.save(npy_path, a1m)

        results.append(
            _measure(
                bench_id="npy_load_8mb",
                category="Binary I/O",
                label="load(.npy) [1M f64, 8 MB]",
                baseline="numpy",
                threads=1,
                check_tol=1e-8,
                checksum_fn=lambda: np.sum(np.load(npy_path)),
                bench_fn=lambda: np.load(npy_path),
                args=args,
            )
        )

        # --- 7. Spatial Distance Metrics (only when SciPy is installed) ---
        if sp_distance is not None:
            dist_a = _uniform_f64((500, 100), 2121)
            dist_b = _uniform_f64((500, 100), 2222)

            results.append(
                _measure(
                    bench_id="cdist_euclidean_500x100",
                    category="Spatial Distance Metrics (vs SciPy)",
                    label="cdist(A, B, euclidean) [500x100 vs 500x100]",
                    baseline="scipy",
                    threads=1,
                    check_tol=1e-6,
                    checksum_fn=lambda: np.sum(
                        sp_distance.cdist(dist_a, dist_b, metric="euclidean")
                    ),
                    bench_fn=lambda: sp_distance.cdist(
                        dist_a, dist_b, metric="euclidean"
                    ),
                    args=args,
                )
            )

            results.append(
                _measure(
                    bench_id="pdist_euclidean_500x100",
                    category="Spatial Distance Metrics (vs SciPy)",
                    label="pdist(A, euclidean) [500x100 f64]",
                    baseline="scipy",
                    threads=1,
                    check_tol=1e-6,
                    checksum_fn=lambda: np.sum(
                        sp_distance.pdist(dist_a, metric="euclidean")
                    ),
                    bench_fn=lambda: sp_distance.pdist(dist_a, metric="euclidean"),
                    args=args,
                )
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def _run_multi(args: argparse.Namespace) -> list[dict]:
    results: list[dict] = []
    mm_a1024_f64 = _uniform_f64((1024, 1024), 2323)
    mm_b1024_f64 = _uniform_f64((1024, 1024), 2424)

    results.append(
        _measure(
            bench_id="matmul_1024_f64_mt",
            category="Multi-Threaded Linear Algebra",
            label=f"matmul(A, B) [1024x1024 f64, {args.threads}T]",
            baseline="numpy",
            threads=args.threads,
            check_tol=1e-6,
            checksum_fn=lambda: np.sum(np.matmul(mm_a1024_f64, mm_b1024_f64)),
            bench_fn=lambda: np.matmul(mm_a1024_f64, mm_b1024_f64),
            args=args,
        )
    )

    mm_a1024_f32 = _uniform_f32((1024, 1024), 2525)
    mm_b1024_f32 = _uniform_f32((1024, 1024), 2626)

    results.append(
        _measure(
            bench_id="matmul_1024_f32_mt",
            category="Multi-Threaded Linear Algebra",
            label=f"matmul(A, B) [1024x1024 f32, {args.threads}T]",
            baseline="numpy",
            threads=args.threads,
            check_tol=1e-2,
            checksum_fn=lambda: np.sum(
                np.matmul(mm_a1024_f32, mm_b1024_f32), dtype=np.float64
            ),
            bench_fn=lambda: np.matmul(mm_a1024_f32, mm_b1024_f32),
            args=args,
        )
    )
    return results


def main() -> None:
    blas_info = _detect_blas_info()
    if _ARGS.probe_blas:
        print(json.dumps(blas_info))
        return

    benchmarks: list[dict] = []
    if _ARGS.mode == "single":
        benchmarks.extend(_run_single(_ARGS))
    elif _ARGS.mode == "multi":
        benchmarks.extend(_run_multi(_ARGS))

    payload = {
        "runtime": "python",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": _SCIPY_VERSION,
        "blas": blas_info,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processors": os.cpu_count() or 1,
        "mode": _ARGS.mode,
        "samples": _ARGS.samples,
        "sample_ms": _ARGS.sample_ms,
        "warmup_ms": _ARGS.warmup_ms,
        "threads": _ARGS.threads,
        "sink": _SINK,
        "benchmarks": benchmarks,
    }

    encoded = json.dumps(payload, indent=2)
    if _ARGS.json_path:
        out_path = pathlib.Path(_ARGS.json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
