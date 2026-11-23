// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ndarray/ndarray.dart';

/// Murmur3 32-bit finalizer (bijection on uint32).
///
/// Matches `readme_numpy.py` bit-for-bit so Dart and NumPy operate on
/// identical non-degenerate pseudorandom inputs and can cross-check results.
int _mix32(int x) {
  var h = x & 0xFFFFFFFF;
  h ^= h >>> 16;
  h = (h * 0x85EBCA6B) & 0xFFFFFFFF;
  h ^= h >>> 13;
  h = (h * 0xC2B2AE35) & 0xFFFFFFFF;
  h ^= h >>> 16;
  return h;
}

Float64List _uniformF64List(int count, int seed) {
  final out = Float64List(count);
  for (var i = 0; i < count; i++) {
    out[i] = (_mix32(i + seed) / 4294967296.0) * 2.0 - 1.0;
  }
  return out;
}

Float32List _uniformF32List(int count, int seed) {
  final out = Float32List(count);
  for (var i = 0; i < count; i++) {
    out[i] = (_mix32(i + seed) / 4294967296.0) * 2.0 - 1.0;
  }
  return out;
}

NDArray<Float64> _makeUniformF64(List<int> shape, int seed) {
  final total = shape.fold<int>(1, (a, b) => a * b);
  return NDArray.fromTypedData(
    _uniformF64List(total, seed),
    shape,
    DType.float64,
  );
}

NDArray<Float32> _makeUniformF32(List<int> shape, int seed) {
  final total = shape.fold<int>(1, (a, b) => a * b);
  return NDArray.fromTypedData(
    _uniformF32List(total, seed),
    shape,
    DType.float32,
  );
}

NDArray<Float64> _makeDiagDominantF64(int n, int seed) {
  final data = _uniformF64List(n * n, seed);
  for (var i = 0; i < n; i++) {
    data[i * n + i] += n.toDouble();
  }
  return NDArray.fromTypedData(data, [n, n], DType.float64);
}

NDArray<Float64> _makeSpdF64(int n, int seed) {
  final raw = _uniformF64List(n * n, seed);
  final spd = Float64List(n * n);
  for (var i = 0; i < n; i++) {
    for (var j = i; j < n; j++) {
      final sym = 0.5 * (raw[i * n + j] + raw[j * n + i]);
      final val = i == j ? sym + n.toDouble() : sym;
      spd[i * n + j] = val;
      spd[j * n + i] = val;
    }
  }
  return NDArray.fromTypedData(spd, [n, n], DType.float64);
}

double _percentile(List<double> sorted, double q) {
  if (sorted.isEmpty) return 0.0;
  if (sorted.length == 1) return sorted.first;
  final pos = q * (sorted.length - 1);
  final lo = pos.floor();
  final hi = pos.ceil();
  if (lo == hi) return sorted[lo];
  final frac = pos - lo;
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

double _sink = 0.0;

final class _BenchConfig {
  final String mode;
  final int samples;
  final int sampleMs;
  final int warmupMs;
  final int threads;
  final String? jsonPath;

  const _BenchConfig({
    required this.mode,
    required this.samples,
    required this.sampleMs,
    required this.warmupMs,
    required this.threads,
    required this.jsonPath,
  });
}

_BenchConfig _parseArgs(List<String> args) {
  var mode = 'single';
  var samples = 21;
  var sampleMs = 50;
  var warmupMs = 200;
  var threads = math.min(4, Platform.numberOfProcessors);
  String? jsonPath;

  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (arg == '--quick') {
      samples = 5;
      sampleMs = 15;
      warmupMs = 50;
    } else if (arg == '--mode' && i + 1 < args.length) {
      mode = args[++i];
    } else if (arg.startsWith('--mode=')) {
      mode = arg.substring('--mode='.length);
    } else if (arg == '--samples' && i + 1 < args.length) {
      samples = int.parse(args[++i]);
    } else if (arg.startsWith('--samples=')) {
      samples = int.parse(arg.substring('--samples='.length));
    } else if (arg == '--sample-ms' && i + 1 < args.length) {
      sampleMs = int.parse(args[++i]);
    } else if (arg.startsWith('--sample-ms=')) {
      sampleMs = int.parse(arg.substring('--sample-ms='.length));
    } else if (arg == '--warmup-ms' && i + 1 < args.length) {
      warmupMs = int.parse(args[++i]);
    } else if (arg.startsWith('--warmup-ms=')) {
      warmupMs = int.parse(arg.substring('--warmup-ms='.length));
    } else if (arg == '--threads' && i + 1 < args.length) {
      threads = int.parse(args[++i]);
    } else if (arg.startsWith('--threads=')) {
      threads = int.parse(arg.substring('--threads='.length));
    } else if (arg == '--json' && i + 1 < args.length) {
      jsonPath = args[++i];
    } else if (arg.startsWith('--json=')) {
      jsonPath = arg.substring('--json='.length);
    } else if (arg == '--help' || arg == '-h') {
      stdout.writeln(
        'Usage: readme_benchmark [--mode single|multi|all] [--threads N]\n'
        '       [--samples N] [--sample-ms MS] [--warmup-ms MS] [--quick]\n'
        '       [--json PATH]',
      );
      exit(0);
    } else {
      stderr.writeln('Unknown argument: $arg');
      exit(2);
    }
  }

  if (mode != 'single' && mode != 'multi' && mode != 'all') {
    stderr.writeln('Invalid --mode "$mode": must be single, multi, or all.');
    exit(2);
  }
  if (samples < 1 || sampleMs < 1 || warmupMs < 1 || threads < 1) {
    stderr.writeln('Numeric options must be positive integers.');
    exit(2);
  }

  return _BenchConfig(
    mode: mode,
    samples: samples,
    sampleMs: sampleMs,
    warmupMs: warmupMs,
    threads: threads,
    jsonPath: jsonPath,
  );
}

Map<String, Object?> _measure({
  required String id,
  required String category,
  required String label,
  required String baseline,
  required int threads,
  required double checkTol,
  required double Function() checksumFn,
  required void Function() benchFn,
  required _BenchConfig cfg,
}) {
  // 1. Prime caches and record deterministic verification checksum once.
  final checksum = checksumFn();
  _sink += checksum.isFinite ? checksum : 0.0;

  // 2. Warmup: run at least cfg.warmupMs and at least 5 calls.
  final warmupTargetUs = cfg.warmupMs * 1000;
  final warmWatch = Stopwatch()..start();
  var warmCalls = 0;
  while (warmCalls < 5 || warmWatch.elapsedMicroseconds < warmupTargetUs) {
    benchFn();
    warmCalls++;
  }
  warmWatch.stop();
  final estCallUs = math.max(warmWatch.elapsedMicroseconds / warmCalls, 0.01);

  // 3. Calibrate batch size so each sample targets cfg.sampleMs.
  final batchSize = math.max(
    1,
    math.min(100000, ((cfg.sampleMs * 1000.0) / estCallUs).round()),
  );

  // 4. Collect independent sample timings.
  final sampleUs = <double>[];
  final sw = Stopwatch();
  for (var s = 0; s < cfg.samples; s++) {
    sw
      ..reset()
      ..start();
    for (var b = 0; b < batchSize; b++) {
      benchFn();
    }
    sw.stop();
    sampleUs.add(sw.elapsedMicroseconds / batchSize);
  }

  final sorted = List<double>.of(sampleUs)..sort();
  final medianUs = _percentile(sorted, 0.50);
  final q1Us = _percentile(sorted, 0.25);
  final q3Us = _percentile(sorted, 0.75);
  final meanUs = sampleUs.reduce((a, b) => a + b) / sampleUs.length;

  stdout.writeln(
    'Dart  | ${label.padRight(46)} | median ${medianUs.toStringAsFixed(2).padLeft(10)} us '
    '(IQR ${(q3Us - q1Us).toStringAsFixed(2).padLeft(8)} us, batch=$batchSize)',
  );

  return <String, Object?>{
    'id': id,
    'category': category,
    'label': label,
    'baseline': baseline,
    'threads': threads,
    'median_us': medianUs,
    'iqr_us': q3Us - q1Us,
    'mean_us': meanUs,
    'min_us': sorted.first,
    'max_us': sorted.last,
    'batch_size': batchSize,
    'samples': cfg.samples,
    'checksum': checksum,
    'check_tol': checkTol,
  };
}

List<Map<String, Object?>> _runSingleThreaded(_BenchConfig cfg) {
  setNumThreads(1);
  final results = <Map<String, Object?>>[];
  final tempDir = Directory.systemTemp.createTempSync('ndarray_readme_bench_');

  try {
    NDArray.scope(() {
      // --- 1. Element-wise & Transcendentals ---
      final a1m = _makeUniformF64([1000000], 101);
      final b1m = _makeUniformF64([1000000], 202);
      final out1m = NDArray.create([1000000], DType.float64);

      results.add(
        _measure(
          id: 'add_inplace_1m',
          category: 'Element-wise & Transcendentals',
          label: 'add(a, b, out: out) [1M f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            add(a1m, b1m, out: out1m);
            final s = sum(out1m);
            final v = s.scalar;
            s.dispose();
            return v;
          },
          benchFn: () => add(a1m, b1m, out: out1m),
          cfg: cfg,
        ),
      );

      results.add(
        _measure(
          id: 'add_alloc_1m',
          category: 'Element-wise & Transcendentals',
          label: 'add(a, b) [1M f64, allocating]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            final res = add(a1m, b1m);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = add(a1m, b1m);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      final x100k = _makeUniformF64([100000], 303);
      final out100k = NDArray.create([100000], DType.float64);

      results.add(
        _measure(
          id: 'sin_100k',
          category: 'Element-wise & Transcendentals',
          label: 'sin(x, out: out) [100k f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-7,
          checksumFn: () {
            sin(x100k, out: out100k);
            final s = sum(out100k);
            final v = s.scalar;
            s.dispose();
            return v;
          },
          benchFn: () => sin(x100k, out: out100k),
          cfg: cfg,
        ),
      );

      results.add(
        _measure(
          id: 'exp_100k',
          category: 'Element-wise & Transcendentals',
          label: 'exp(x, out: out) [100k f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-7,
          checksumFn: () {
            exp(x100k, out: out100k);
            final s = sum(out100k);
            final v = s.scalar;
            s.dispose();
            return v;
          },
          benchFn: () => exp(x100k, out: out100k),
          cfg: cfg,
        ),
      );

      final matA1000 = _makeUniformF64([1000, 1000], 404);
      final matB1000 = _makeUniformF64([1000, 1000], 505);
      final matAT = matA1000.transposed;
      final matBT = matB1000.transposed;
      final outMat1000 = NDArray.create([1000, 1000], DType.float64);

      results.add(
        _measure(
          id: 'strided_add_1000x1000',
          category: 'Element-wise & Transcendentals',
          label: 'add(a.T, b.T, out: out) [1000x1000 f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            add(matAT, matBT, out: outMat1000);
            final s = sum(outMat1000);
            final v = s.scalar;
            s.dispose();
            return v;
          },
          benchFn: () => add(matAT, matBT, out: outMat1000),
          cfg: cfg,
        ),
      );

      // --- 2. Reductions & Statistics ---
      results.add(
        _measure(
          id: 'sum_1m',
          category: 'Reductions & Statistics',
          label: 'sum(x) [1M f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            final res = sum(a1m);
            final v = res.scalar;
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = sum(a1m);
            _sink += res.scalar;
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      results.add(
        _measure(
          id: 'mean_axis0_1000x1000',
          category: 'Reductions & Statistics',
          label: 'mean(mat, axis: 0) [1000x1000 f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            final res = mean(matA1000, axis: 0);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = mean(matA1000, axis: 0);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      final nanData = _uniformF64List(100000, 606);
      for (var i = 0; i < nanData.length; i += 10) {
        nanData[i] = double.nan;
      }
      final nanVec = NDArray.fromTypedData(nanData, [100000], DType.float64);

      results.add(
        _measure(
          id: 'nansum_100k',
          category: 'Reductions & Statistics',
          label: 'nansum(x) [100k f64, 10% NaN]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            final res = nansum(nanVec);
            final v = res.scalar;
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = nansum(nanVec);
            _sink += res.scalar;
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      // --- 3. Linear Algebra & Tensor Contractions ---
      final mmA256 = _makeUniformF64([256, 256], 707);
      final mmB256 = _makeUniformF64([256, 256], 808);

      results.add(
        _measure(
          id: 'matmul_256_f64',
          category: 'Linear Algebra & Tensor Contractions',
          label: 'matmul(A, B) [256x256 f64, 1T]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-7,
          checksumFn: () {
            final res = matmul(mmA256, mmB256);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = matmul(mmA256, mmB256);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      final mmA512 = _makeUniformF32([512, 512], 909);
      final mmB512 = _makeUniformF32([512, 512], 1010);

      results.add(
        _measure(
          id: 'matmul_512_f32',
          category: 'Linear Algebra & Tensor Contractions',
          label: 'matmul(A, B) [512x512 f32, 1T]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-3,
          checksumFn: () {
            final res = matmul(mmA512, mmB512);
            final s = sumAs(res, DType.float64);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = matmul(mmA512, mmB512);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      final solveA = _makeDiagDominantF64(200, 1111);
      final solveB = _makeUniformF64([200, 1], 1212);

      results.add(
        _measure(
          id: 'solve_200_f64',
          category: 'Linear Algebra & Tensor Contractions',
          label: 'solve(A, b) [200x200 f64, 1T]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            final res = solve(solveA, solveB);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = solve(solveA, solveB);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      final svdA = _makeUniformF64([128, 128], 1313);

      results.add(
        _measure(
          id: 'svd_128_f64',
          category: 'Linear Algebra & Tensor Contractions',
          label: 'svd(A, fullMatrices: false) [128x128 f64, 1T]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-7,
          checksumFn: () {
            final res = svd(svdA, fullMatrices: false);
            final s = sum(res.s);
            final v = s.scalar;
            s.dispose();
            res.u.dispose();
            res.s.dispose();
            res.vh.dispose();
            return v;
          },
          benchFn: () {
            final res = svd(svdA, fullMatrices: false);
            res.u.dispose();
            res.s.dispose();
            res.vh.dispose();
          },
          cfg: cfg,
        ),
      );

      final spdA = _makeSpdF64(200, 1414);

      results.add(
        _measure(
          id: 'cholesky_200_f64',
          category: 'Linear Algebra & Tensor Contractions',
          label: 'cholesky(A) [200x200 SPD f64, 1T]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-7,
          checksumFn: () {
            final res = cholesky(spdA);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = cholesky(spdA);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      final attnQ = _makeUniformF32([2, 8, 64, 64], 1515);
      final attnK = _makeUniformF32([2, 8, 64, 64], 1616);
      final attnSubs = EinsumSubscripts.parse('bhid,bhjd->bhij');

      results.add(
        _measure(
          id: 'einsum_attn_f32',
          category: 'Linear Algebra & Tensor Contractions',
          label: "einsum('bhid,bhjd->bhij') [2x8x64x64 f32, 1T]",
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-3,
          checksumFn: () {
            final res = einsum(attnSubs, [attnQ, attnK]);
            final s = sumAs(res, DType.float64);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = einsum(attnSubs, [attnQ, attnK]);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      // --- 4. Spectral Transforms & Complex Math ---
      final rfftSig = _makeUniformF64([65536], 1717);

      results.add(
        _measure(
          id: 'rfft_65536_f64',
          category: 'Spectral Transforms & Complex Math',
          label: 'rfft(x) [65,536 f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-5,
          checksumFn: () {
            final spec = rfft(rfftSig);
            final mag = abs(spec);
            final s = sum(mag);
            final v = s.scalar;
            s.dispose();
            mag.dispose();
            spec.dispose();
            return v;
          },
          benchFn: () {
            final spec = rfft(rfftSig);
            spec.dispose();
          },
          cfg: cfg,
        ),
      );

      final img512 = _makeUniformF64([512, 512], 1818);

      results.add(
        _measure(
          id: 'fft2_512_f64',
          category: 'Spectral Transforms & Complex Math',
          label: 'fft2(img) [512x512 f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-4,
          checksumFn: () {
            final spec = fft2(img512);
            final mag = abs(spec);
            final s = sum(mag);
            final v = s.scalar;
            s.dispose();
            mag.dispose();
            spec.dispose();
            return v;
          },
          benchFn: () {
            final spec = fft2(img512);
            spec.dispose();
          },
          cfg: cfg,
        ),
      );

      final c100k = NDArray.fromTypedData(_uniformF64List(200000, 1919), [
        100000,
      ], DType.complex128);

      results.add(
        _measure(
          id: 'complex_abs_100k',
          category: 'Spectral Transforms & Complex Math',
          label: 'abs(z) [100k complex128 -> f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-7,
          checksumFn: () {
            final res = abs(c100k);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = abs(c100k);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      // --- 5. Sorting & Selection ---
      final sortVec = _makeUniformF64([100000], 2020);

      results.add(
        _measure(
          id: 'sort_100k_f64',
          category: 'Sorting & Selection',
          label: 'sort(x) [100k random f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-12,
          checksumFn: () {
            final res = sort(sortVec);
            final v =
                res.getCell([0]) + res.getCell([50000]) + res.getCell([99999]);
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = sort(sortVec);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      results.add(
        _measure(
          id: 'argsort_100k_f64',
          category: 'Sorting & Selection',
          label: 'argsort(x) [100k random f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-12,
          checksumFn: () {
            final res = argsort(sortVec);
            final v =
                (res.getCell([0]) + res.getCell([50000]) + res.getCell([99999]))
                    .toDouble();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = argsort(sortVec);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      results.add(
        _measure(
          id: 'partition_100k_f64',
          category: 'Sorting & Selection',
          label: 'partition(x, 50000) [100k random f64]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-12,
          checksumFn: () {
            final res = partition(sortVec, 50000);
            final v = res.getCell([50000]);
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = partition(sortVec, 50000);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      // --- 6. Binary I/O (.npy) ---
      final npyPath = '${tempDir.path}/array_1m.npy';
      save(npyPath, a1m);

      results.add(
        _measure(
          id: 'npy_load_8mb',
          category: 'Binary I/O',
          label: 'load(.npy) [1M f64, 8 MB]',
          baseline: 'numpy',
          threads: 1,
          checkTol: 1e-8,
          checksumFn: () {
            final res = load(npyPath) as NDArray<Float64>;
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = load(npyPath);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      // --- 7. Spatial Distance Metrics (paired against SciPy when available) ---
      final distA = _makeUniformF64([500, 100], 2121);
      final distB = _makeUniformF64([500, 100], 2222);

      results.add(
        _measure(
          id: 'cdist_euclidean_500x100',
          category: 'Spatial Distance Metrics (vs SciPy)',
          label: 'cdist(A, B, euclidean) [500x100 vs 500x100]',
          baseline: 'scipy',
          threads: 1,
          checkTol: 1e-6,
          checksumFn: () {
            final res = cdist(distA, distB, metric: DistanceMetric.euclidean);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = cdist(distA, distB, metric: DistanceMetric.euclidean);
            res.dispose();
          },
          cfg: cfg,
        ),
      );

      results.add(
        _measure(
          id: 'pdist_euclidean_500x100',
          category: 'Spatial Distance Metrics (vs SciPy)',
          label: 'pdist(A, euclidean) [500x100 f64]',
          baseline: 'scipy',
          threads: 1,
          checkTol: 1e-6,
          checksumFn: () {
            final res = pdist(distA, metric: DistanceMetric.euclidean);
            final s = sum(res);
            final v = s.scalar;
            s.dispose();
            res.dispose();
            return v;
          },
          benchFn: () {
            final res = pdist(distA, metric: DistanceMetric.euclidean);
            res.dispose();
          },
          cfg: cfg,
        ),
      );
    });
  } finally {
    if (tempDir.existsSync()) {
      tempDir.deleteSync(recursive: true);
    }
  }

  return results;
}

List<Map<String, Object?>> _runMultiThreaded(_BenchConfig cfg) {
  setNumThreads(cfg.threads);
  final results = <Map<String, Object?>>[];

  NDArray.scope(() {
    final mmA1024F64 = _makeUniformF64([1024, 1024], 2323);
    final mmB1024F64 = _makeUniformF64([1024, 1024], 2424);

    results.add(
      _measure(
        id: 'matmul_1024_f64_mt',
        category: 'Multi-Threaded Linear Algebra',
        label: 'matmul(A, B) [1024x1024 f64, ${cfg.threads}T]',
        baseline: 'numpy',
        threads: cfg.threads,
        checkTol: 1e-6,
        checksumFn: () {
          final res = matmul(mmA1024F64, mmB1024F64);
          final s = sum(res);
          final v = s.scalar;
          s.dispose();
          res.dispose();
          return v;
        },
        benchFn: () {
          final res = matmul(mmA1024F64, mmB1024F64);
          res.dispose();
        },
        cfg: cfg,
      ),
    );

    final mmA1024F32 = _makeUniformF32([1024, 1024], 2525);
    final mmB1024F32 = _makeUniformF32([1024, 1024], 2626);

    results.add(
      _measure(
        id: 'matmul_1024_f32_mt',
        category: 'Multi-Threaded Linear Algebra',
        label: 'matmul(A, B) [1024x1024 f32, ${cfg.threads}T]',
        baseline: 'numpy',
        threads: cfg.threads,
        checkTol: 1e-2,
        checksumFn: () {
          final res = matmul(mmA1024F32, mmB1024F32);
          final s = sumAs(res, DType.float64);
          final v = s.scalar;
          s.dispose();
          res.dispose();
          return v;
        },
        benchFn: () {
          final res = matmul(mmA1024F32, mmB1024F32);
          res.dispose();
        },
        cfg: cfg,
      ),
    );
  });

  return results;
}

void main(List<String> args) {
  final cfg = _parseArgs(args);
  final benchmarks = <Map<String, Object?>>[];

  if (cfg.mode == 'single' || cfg.mode == 'all') {
    benchmarks.addAll(_runSingleThreaded(cfg));
  }
  if (cfg.mode == 'multi' || cfg.mode == 'all') {
    benchmarks.addAll(_runMultiThreaded(cfg));
  }

  final payload = <String, Object?>{
    'runtime': 'dart',
    'dart_version': Platform.version,
    'os': Platform.operatingSystem,
    'os_version': Platform.operatingSystemVersion,
    'processors': Platform.numberOfProcessors,
    'mode': cfg.mode,
    'samples': cfg.samples,
    'sample_ms': cfg.sampleMs,
    'warmup_ms': cfg.warmupMs,
    'threads': cfg.threads,
    'sink': _sink,
    'benchmarks': benchmarks,
  };

  final encoded = const JsonEncoder.withIndent('  ').convert(payload);
  if (cfg.jsonPath != null) {
    final file = File(cfg.jsonPath!);
    file.parent.createSync(recursive: true);
    file.writeAsStringSync(encoded);
  }
}
