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
import 'dart:ffi' show Abi;
import 'dart:io';
import 'dart:math' as math;

/// Curated AOT (`dart build cli`) vs NumPy/SciPy README benchmark orchestrator.
///
/// Run from `pkgs/ndarray`:
/// ```bash
/// dart run benchmark/run_readme_benchmark.dart
/// ```
///
/// Compiles `benchmark/readme_benchmark.dart` to an AOT native CLI bundle
/// (including `libndarray`, `libopenblas`, and `libpocketfft` code assets),
/// executes both single-threaded and multi-threaded runs alongside
/// `benchmark/readme_numpy.py`, cross-checks numerical checksums, flags Netlib
/// reference BLAS if detected on Python, and emits Markdown + JSON reports.
final class _RunnerOptions {
  final String pythonExecutable;
  final int samples;
  final int sampleMs;
  final int warmupMs;
  final int threads;
  final bool quick;
  final bool skipMulti;
  final bool strictBlas;
  final String markdownOut;
  final String jsonOut;

  const _RunnerOptions({
    required this.pythonExecutable,
    required this.samples,
    required this.sampleMs,
    required this.warmupMs,
    required this.threads,
    required this.quick,
    required this.skipMulti,
    required this.strictBlas,
    required this.markdownOut,
    required this.jsonOut,
  });
}

_RunnerOptions _parseOptions(List<String> args) {
  var pythonExecutable = Platform.isWindows ? 'python' : 'python3';
  var samples = 21;
  var sampleMs = 50;
  var warmupMs = 200;
  var threads = math.min(4, Platform.numberOfProcessors);
  var quick = false;
  var skipMulti = false;
  var strictBlas = false;
  var markdownOut = 'benchmark/report/readme_comparison.md';
  var jsonOut = 'benchmark/report/readme_comparison.json';

  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (arg == '--python' && i + 1 < args.length) {
      pythonExecutable = args[++i];
    } else if (arg.startsWith('--python=')) {
      pythonExecutable = arg.substring('--python='.length);
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
    } else if (arg == '--quick') {
      quick = true;
      samples = 5;
      sampleMs = 15;
      warmupMs = 50;
    } else if (arg == '--skip-multi') {
      skipMulti = true;
    } else if (arg == '--strict-blas') {
      strictBlas = true;
    } else if (arg == '--markdown' && i + 1 < args.length) {
      markdownOut = args[++i];
    } else if (arg.startsWith('--markdown=')) {
      markdownOut = arg.substring('--markdown='.length);
    } else if (arg == '--json' && i + 1 < args.length) {
      jsonOut = args[++i];
    } else if (arg.startsWith('--json=')) {
      jsonOut = arg.substring('--json='.length);
    } else if (arg == '--help' || arg == '-h') {
      stdout.writeln(
        'Usage: dart run benchmark/run_readme_benchmark.dart [options]\n'
        '  --python PATH       Python interpreter with NumPy (+ optional SciPy)\n'
        '  --samples N         Samples per benchmark (default: 21)\n'
        '  --sample-ms MS      Target ms per sample batch (default: 50)\n'
        '  --warmup-ms MS      Minimum warmup ms per benchmark (default: 200)\n'
        '  --threads N         Thread count for multi-threaded matmul (default: min(4, cores))\n'
        '  --quick             Fast smoke run (5 samples, 15ms/sample, 50ms warmup)\n'
        '  --skip-multi        Skip multi-threaded matmul benchmarks\n'
        '  --strict-blas       Fail if Python NumPy links Netlib reference BLAS\n'
        '  --markdown PATH     Output Markdown report path\n'
        '  --json PATH         Output JSON report path',
      );
      exit(0);
    } else {
      stderr.writeln('Unknown argument: $arg');
      exit(2);
    }
  }

  return _RunnerOptions(
    pythonExecutable: pythonExecutable,
    samples: samples,
    sampleMs: sampleMs,
    warmupMs: warmupMs,
    threads: threads,
    quick: quick,
    skipMulti: skipMulti,
    strictBlas: strictBlas,
    markdownOut: markdownOut,
    jsonOut: jsonOut,
  );
}

String _detectCpuModel() {
  try {
    if (Platform.isLinux) {
      final cpuinfo = File('/proc/cpuinfo').readAsLinesSync();
      for (final line in cpuinfo) {
        if (line.startsWith('model name')) {
          final idx = line.indexOf(':');
          if (idx >= 0) return line.substring(idx + 1).trim();
        }
      }
    } else if (Platform.isMacOS) {
      final res = Process.runSync('sysctl', ['-n', 'machdep.cpu.brand_string']);
      if (res.exitCode == 0) {
        final s = (res.stdout as String).trim();
        if (s.isNotEmpty) return s;
      }
    }
  } catch (_) {}
  return '${Platform.operatingSystem} (${Platform.numberOfProcessors} logical CPUs)';
}

String _detectSimdBaseline() => switch (Abi.current()) {
  Abi.linuxX64 || Abi.macosX64 || Abi.windowsX64 => 'x86-64-v3 AVX2+FMA+F16C',
  Abi.linuxArm64 || Abi.macosArm64 || Abi.windowsArm64 => 'ARM64 NEON',
  _ => Abi.current().toString(),
};

String _formatLatency(double us) {
  if (us >= 1000.0) {
    return '${(us / 1000.0).toStringAsFixed(2)} ms';
  }
  return '${us.toStringAsFixed(1)} µs';
}

String _formatSpeedup(double ratio) {
  if (ratio >= 1.0) {
    return '**${ratio.toStringAsFixed(2)}x**';
  }
  return '${ratio.toStringAsFixed(2)}x';
}

double _geomean(Iterable<double> values) {
  final valid = values.where((v) => v > 0 && v.isFinite).toList();
  if (valid.isEmpty) return 1.0;
  var logSum = 0.0;
  for (final v in valid) {
    logSum += math.log(v);
  }
  return math.exp(logSum / valid.length);
}

final class _BenchmarkExitException implements Exception {
  final int code;
  const _BenchmarkExitException(this.code);
}

Future<void> _runChecked(
  String executable,
  List<String> args, {
  Map<String, String>? environment,
  required String description,
}) async {
  stdout.writeln('\n==> $description');
  final proc = await Process.start(
    executable,
    args,
    environment: environment,
    mode: ProcessStartMode.inheritStdio,
  );
  final code = await proc.exitCode;
  if (code != 0) {
    stderr.writeln('ERROR: "$description" failed with exit code $code.');
    throw _BenchmarkExitException(code);
  }
}

Future<Map<String, dynamic>> _probePythonBlas(String pythonExecutable) async {
  final res = await Process.run(pythonExecutable, const [
    'benchmark/readme_numpy.py',
    '--probe-blas',
  ], environment: _threadEnv(1));
  if (res.exitCode != 0) {
    stderr.writeln(
      'ERROR: Failed to probe Python NumPy BLAS via `$pythonExecutable` '
      '(exit ${res.exitCode}):\n${res.stderr}',
    );
    throw _BenchmarkExitException(res.exitCode);
  }
  return jsonDecode((res.stdout as String).trim()) as Map<String, dynamic>;
}

Map<String, String> _threadEnv(int threads) => <String, String>{
  ...Platform.environment,
  'OPENBLAS_NUM_THREADS': '$threads',
  'MKL_NUM_THREADS': '$threads',
  'OMP_NUM_THREADS': '$threads',
  'VECLIB_MAXIMUM_THREADS': '$threads',
  'NUMEXPR_NUM_THREADS': '$threads',
  'BLIS_NUM_THREADS': '$threads',
};

Future<void> main(List<String> args) async {
  final opts = _parseOptions(args);

  if (!File('benchmark/readme_benchmark.dart').existsSync()) {
    stderr.writeln(
      'Run this script from pkgs/ndarray (missing benchmark/readme_benchmark.dart).',
    );
    exitCode = 2;
    return;
  }

  Directory? workDir;
  try {
    final probeBlas = await _probePythonBlas(opts.pythonExecutable);
    if (probeBlas['is_reference_netlib'] == true) {
      final probeLib = (probeBlas['blas_lib'] as String?) ?? 'unknown';
      stderr.writeln(
        '\nWARNING: Python NumPy links Netlib reference BLAS ($probeLib).\n'
        '         BLAS/LAPACK comparisons will heavily favor ndarray (OpenBLAS).\n'
        '         Use PyPI wheel NumPy (bundled OpenBLAS) for publishable numbers.',
      );
      if (opts.strictBlas) {
        stderr.writeln(
          'ERROR: --strict-blas set; refusing to accept Netlib BLAS.',
        );
        throw const _BenchmarkExitException(3);
      }
    }

    workDir = Directory.systemTemp.createTempSync('ndarray_aot_bench_');
    final buildOutDir = '${workDir.path}/aot_build';
    await _runChecked(Platform.resolvedExecutable, [
      'build',
      'cli',
      '--target',
      'benchmark/readme_benchmark.dart',
      '--output',
      buildOutDir,
    ], description: 'Building AOT CLI bundle via `dart build cli`');

    final exeName = Platform.isWindows
        ? 'readme_benchmark.exe'
        : 'readme_benchmark';
    final aotBinary = '$buildOutDir/bundle/bin/$exeName';
    if (!File(aotBinary).existsSync()) {
      stderr.writeln('ERROR: Expected AOT binary at $aotBinary');
      throw const _BenchmarkExitException(1);
    }

    final commonFlags = <String>[
      '--samples',
      '${opts.samples}',
      '--sample-ms',
      '${opts.sampleMs}',
      '--warmup-ms',
      '${opts.warmupMs}',
      '--threads',
      '${opts.threads}',
    ];

    final dartSingleJson = '${workDir.path}/dart_single.json';
    final pySingleJson = '${workDir.path}/py_single.json';
    await _runChecked(
      aotBinary,
      ['--mode', 'single', ...commonFlags, '--json', dartSingleJson],
      environment: _threadEnv(1),
      description: 'Running Dart AOT single-threaded benchmarks',
    );
    await _runChecked(
      opts.pythonExecutable,
      [
        'benchmark/readme_numpy.py',
        '--mode',
        'single',
        ...commonFlags,
        '--json',
        pySingleJson,
      ],
      environment: _threadEnv(1),
      description: 'Running Python NumPy/SciPy single-threaded benchmarks',
    );

    final dartPayload =
        jsonDecode(File(dartSingleJson).readAsStringSync())
            as Map<String, dynamic>;
    final pyPayload =
        jsonDecode(File(pySingleJson).readAsStringSync())
            as Map<String, dynamic>;

    final dartEntries = <Map<String, dynamic>>[
      ...(dartPayload['benchmarks'] as List<dynamic>)
          .cast<Map<String, dynamic>>(),
    ];
    final pyEntries = <Map<String, dynamic>>[
      ...(pyPayload['benchmarks'] as List<dynamic>)
          .cast<Map<String, dynamic>>(),
    ];

    if (!opts.skipMulti && opts.threads > 1) {
      final dartMultiJson = '${workDir.path}/dart_multi.json';
      final pyMultiJson = '${workDir.path}/py_multi.json';
      await _runChecked(
        aotBinary,
        ['--mode', 'multi', ...commonFlags, '--json', dartMultiJson],
        environment: _threadEnv(opts.threads),
        description:
            'Running Dart AOT multi-threaded benchmarks (${opts.threads}T)',
      );
      await _runChecked(
        opts.pythonExecutable,
        [
          'benchmark/readme_numpy.py',
          '--mode',
          'multi',
          ...commonFlags,
          '--json',
          pyMultiJson,
        ],
        environment: _threadEnv(opts.threads),
        description:
            'Running Python NumPy multi-threaded benchmarks (${opts.threads}T)',
      );
      final dm =
          jsonDecode(File(dartMultiJson).readAsStringSync())
              as Map<String, dynamic>;
      final pm =
          jsonDecode(File(pyMultiJson).readAsStringSync())
              as Map<String, dynamic>;
      dartEntries.addAll(
        (dm['benchmarks'] as List<dynamic>).cast<Map<String, dynamic>>(),
      );
      pyEntries.addAll(
        (pm['benchmarks'] as List<dynamic>).cast<Map<String, dynamic>>(),
      );
    }

    final blasInfo = (pyPayload['blas'] as Map<String, dynamic>?) ?? const {};
    final isReferenceNetlib = blasInfo['is_reference_netlib'] == true;
    final blasName = (blasInfo['blas_name'] as String?) ?? 'unknown';
    final blasLib = (blasInfo['blas_lib'] as String?) ?? 'unknown';

    if (isReferenceNetlib && opts.strictBlas) {
      stderr.writeln(
        'ERROR: --strict-blas set; refusing to accept Netlib BLAS ($blasLib).',
      );
      throw const _BenchmarkExitException(3);
    }

    final pyById = <String, Map<String, dynamic>>{
      for (final e in pyEntries) e['id'] as String: e,
    };

    final pairedRows = <Map<String, Object?>>[];
    final checksumFailures = <String>[];

    for (final d in dartEntries) {
      final id = d['id'] as String;
      final p = pyById[id];
      if (p == null) continue;

      final dMed = (d['median_us'] as num).toDouble();
      final dIqr = (d['iqr_us'] as num).toDouble();
      final pMed = (p['median_us'] as num).toDouble();
      final pIqr = (p['iqr_us'] as num).toDouble();
      final dCheck = (d['checksum'] as num).toDouble();
      final pCheck = (p['checksum'] as num).toDouble();
      final tol = (d['check_tol'] as num).toDouble();

      final scale = math.max(1.0, math.max(dCheck.abs(), pCheck.abs()));
      final relErr = (dCheck - pCheck).abs() / scale;
      if (!relErr.isFinite || relErr > tol) {
        checksumFailures.add(
          '$id: Dart=$dCheck vs Python=$pCheck (relErr=$relErr > tol=$tol)',
        );
      }

      pairedRows.add(<String, Object?>{
        'id': id,
        'category': d['category'],
        'label': d['label'],
        'baseline': d['baseline'],
        'threads': d['threads'],
        'dart_median_us': dMed,
        'dart_iqr_us': dIqr,
        'python_median_us': pMed,
        'python_iqr_us': pIqr,
        'speedup': pMed / dMed,
        'checksum_rel_err': relErr,
      });
    }

    if (checksumFailures.isNotEmpty) {
      stderr.writeln(
        '\nERROR: Numerical parity check failed between Dart and Python:\n'
        '  ${checksumFailures.join('\n  ')}',
      );
      throw const _BenchmarkExitException(4);
    }

    final numpyRows = pairedRows
        .where((r) => r['baseline'] == 'numpy' && r['threads'] == 1)
        .toList();
    final scipyRows = pairedRows
        .where((r) => r['baseline'] == 'scipy')
        .toList();
    final numpyGeomean = _geomean(
      numpyRows.map((r) => (r['speedup'] as num).toDouble()),
    );

    final cpuModel = _detectCpuModel();
    final dartVersion = Platform.version.split(' ').first;
    final numpyVersion = pyPayload['numpy_version'] as String;
    final scipyVersion = pyPayload['scipy_version'] as String?;

    final md = StringBuffer()
      ..writeln('# `package:ndarray` (Dart AOT) vs NumPy Benchmarks')
      ..writeln();

    if (isReferenceNetlib) {
      md
        ..writeln('> [!WARNING]')
        ..writeln(
          '> Python NumPy in this run links **Netlib reference BLAS** (`$blasLib`) rather than OpenBLAS/MKL. '
          'Linear algebra rows (`matmul`, `solve`, `svd`, `cholesky`, `einsum`) are NOT apples-to-apples; '
          're-run with PyPI wheel NumPy before publishing.',
        )
        ..writeln();
    }

    md
      ..writeln('## Environment & Methodology')
      ..writeln()
      ..writeln(
        '- **CPU:** $cpuModel (${Platform.numberOfProcessors} logical cores)',
      )
      ..writeln(
        '- **OS:** ${Platform.operatingSystem} (${Platform.operatingSystemVersion})',
      )
      ..writeln(
        '- **Dart:** `$dartVersion` (AOT native bundle via `dart build cli`, ${_detectSimdBaseline()})',
      )
      ..writeln(
        '- **Python / NumPy:** Python `${pyPayload['python_version']}`, NumPy `$numpyVersion` (BLAS: `$blasName`)',
      )
      ..writeln(
        '- **SciPy:** ${scipyVersion != null ? '`$scipyVersion`' : 'not installed (`cdist`/`pdist` rows skipped)'}',
      )
      ..writeln(
        '- **Timing:** Identical warmup (${opts.warmupMs} ms min) + ${opts.samples} batch-calibrated samples (~${opts.sampleMs} ms/sample), reporting **median** latency (and IQR). '
        'All ${pairedRows.length} benchmarks passed numerical parity checksum verification against NumPy${scipyVersion != null ? '/SciPy' : ''}.',
      )
      ..writeln(
        isReferenceNetlib
            ? '- **Single-threaded NumPy Geomean (${numpyRows.length} ops):** *suppressed (Python links Netlib reference BLAS; re-run with OpenBLAS/MKL)*'
            : '- **Single-threaded NumPy Geomean (${numpyRows.length} ops):** **${numpyGeomean.toStringAsFixed(2)}x** (`> 1.0x` means `ndarray` is faster)',
      )
      ..writeln()
      ..writeln('## Results')
      ..writeln()
      ..writeln(
        '| Category | Operation | `ndarray` AOT (median) | NumPy / SciPy (median) | Relative (`ndarray` vs baseline) |',
      )
      ..writeln('| :--- | :--- | ---: | ---: | ---: |');

    var lastCategory = '';
    for (final r in pairedRows) {
      final cat = r['category'] as String;
      final catCell = cat == lastCategory ? '' : '**$cat**';
      lastCategory = cat;
      final label = r['label'] as String;
      final dMed = (r['dart_median_us'] as num).toDouble();
      final pMed = (r['python_median_us'] as num).toDouble();
      final speedup = (r['speedup'] as num).toDouble();
      final baselineTag = r['baseline'] == 'scipy' ? ' *(SciPy)*' : '';
      md.writeln(
        '| $catCell | `$label` | ${_formatLatency(dMed)} | ${_formatLatency(pMed)}$baselineTag | ${_formatSpeedup(speedup)} |',
      );
    }

    if (scipyRows.isEmpty) {
      md
        ..writeln()
        ..writeln(
          '*Note: `cdist` / `pdist` rows are omitted because `scipy` was not installed in `${opts.pythonExecutable}`.*',
        );
    }

    final mdFile = File(opts.markdownOut);
    mdFile.parent.createSync(recursive: true);
    mdFile.writeAsStringSync(md.toString());

    final jsonFile = File(opts.jsonOut);
    jsonFile.parent.createSync(recursive: true);
    jsonFile.writeAsStringSync(
      const JsonEncoder.withIndent('  ').convert(<String, Object?>{
        'cpu': cpuModel,
        'os': Platform.operatingSystem,
        'os_version': Platform.operatingSystemVersion,
        'processors': Platform.numberOfProcessors,
        'dart_version': Platform.version,
        'python_version': pyPayload['python_version'],
        'numpy_version': numpyVersion,
        'scipy_version': scipyVersion,
        'blas': blasInfo,
        'samples': opts.samples,
        'sample_ms': opts.sampleMs,
        'warmup_ms': opts.warmupMs,
        'threads': opts.threads,
        'single_thread_numpy_geomean': numpyGeomean,
        'benchmarks': pairedRows,
      }),
    );

    stdout.writeln('\n$md');
    stdout.writeln('Wrote Markdown report: ${mdFile.path}');
    stdout.writeln('Wrote JSON summary:    ${jsonFile.path}');
  } on _BenchmarkExitException catch (e) {
    exitCode = e.code;
  } finally {
    if (workDir != null && workDir.existsSync()) {
      workDir.deleteSync(recursive: true);
    }
  }
}
