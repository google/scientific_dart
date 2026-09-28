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

/// Describes a paired Dart (`package:ndarray`) and Python (`NumPy`) benchmark suite.
final class BenchmarkSuiteSpec {
  final String id;
  final String title;
  final String dartFile;
  final String pythonFile;

  const BenchmarkSuiteSpec({
    required this.id,
    required this.title,
    required this.dartFile,
    required this.pythonFile,
  });
}

const List<BenchmarkSuiteSpec> allSuites = [
  BenchmarkSuiteSpec(
    id: 'master',
    title: 'Master All-Inclusive Suite',
    dartFile: 'benchmark/perf_benchmarks.dart',
    pythonFile: 'benchmark/numpy_benchmarks.py',
  ),
  BenchmarkSuiteSpec(
    id: 'stats_reductions',
    title: 'Statistics & Reductions',
    dartFile: 'benchmark/stats_reductions_benchmark.dart',
    pythonFile: 'benchmark/numpy_stats_reductions_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'linalg_solvers',
    title: 'Linear Algebra Solvers & Invariants',
    dartFile: 'benchmark/linalg_solvers_benchmark.dart',
    pythonFile: 'benchmark/numpy_linalg_solvers_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'eigen_selection',
    title: 'Eigenvalues, Matrix Chains, Partitioning & Search',
    dartFile: 'benchmark/eigen_selection_benchmark.dart',
    pythonFile: 'benchmark/numpy_eigen_selection_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'nan_cumulative_grids',
    title: 'NaN Reductions, Cumulative Scans, Floating-Point & Grids',
    dartFile: 'benchmark/nan_cumulative_grids_benchmark.dart',
    pythonFile: 'benchmark/numpy_nan_cumulative_grids_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'fft_dsp',
    title: 'Real FFT, 2D FFT & DSP Windows',
    dartFile: 'benchmark/fft_dsp_benchmark.dart',
    pythonFile: 'benchmark/numpy_fft_dsp_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'indexing_manipulation',
    title: 'Indexing, Slicing & Array Manipulation',
    dartFile: 'benchmark/indexing_manipulation_benchmark.dart',
    pythonFile: 'benchmark/numpy_indexing_manipulation_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'binning_set_ops',
    title: 'Binning, Histograms & Set Operations',
    dartFile: 'benchmark/binning_set_ops_benchmark.dart',
    pythonFile: 'benchmark/numpy_binning_set_ops_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'bitwise_windows',
    title: 'Bitwise Operations, Windows & Special Functions',
    dartFile: 'benchmark/bitwise_windows_benchmark.dart',
    pythonFile: 'benchmark/numpy_bitwise_windows_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'complex_math',
    title: 'Complex128 Vectorized Math',
    dartFile: 'benchmark/complex_math_benchmark.dart',
    pythonFile: 'benchmark/numpy_complex_math_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'padding_transforms',
    title: 'Padding, Flipping, Rolling & Splitting',
    dartFile: 'benchmark/padding_transforms_benchmark.dart',
    pythonFile: 'benchmark/numpy_padding_transforms_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'polynomial_interp',
    title: 'Polynomial Fitting, Chebyshev & 1D Interpolation',
    dartFile: 'benchmark/polynomial_interp_benchmark.dart',
    pythonFile: 'benchmark/numpy_polynomial_interp_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'random_distributions',
    title: 'Random Number Generation & Distributions',
    dartFile: 'benchmark/random_distributions_benchmark.dart',
    pythonFile: 'benchmark/numpy_random_distributions_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'spatial_contractions',
    title: 'Spatial Distance Metrics & Tensor Contractions',
    dartFile: 'benchmark/spatial_contractions_benchmark.dart',
    pythonFile: 'benchmark/numpy_spatial_contractions_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'ufunc_reductions',
    title: 'Ufunc Reductions, Accumulate, Outer, Reduceat & Masked Ops',
    dartFile: 'benchmark/ufunc_reductions_benchmark.dart',
    pythonFile: 'benchmark/numpy_ufunc_reductions_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'operations',
    title: 'Einsum, Tensordot, Convolution & Correlation',
    dartFile: 'benchmark/operations_benchmark.dart',
    pythonFile: 'benchmark/numpy_operations_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'sort',
    title: 'Sorting & Argsort Across Input Distributions',
    dartFile: 'benchmark/sort_benchmark.dart',
    pythonFile: 'benchmark/numpy_sort_benchmarks.py',
  ),
  BenchmarkSuiteSpec(
    id: 'calculus_integration',
    title: 'Numerical Differentiation & Integration',
    dartFile: 'benchmark/calculus_integration_benchmark.dart',
    pythonFile: 'benchmark/numpy_calculus_integration_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'calculus',
    title: '1M-Element Calculus (trapz & gradient)',
    dartFile: 'benchmark/calculus_benchmark.dart',
    pythonFile: 'benchmark/numpy_calculus_benchmarks.py',
  ),
  BenchmarkSuiteSpec(
    id: 'financial',
    title: 'Quantitative Financial Operations',
    dartFile: 'benchmark/financial_benchmark.dart',
    pythonFile: 'benchmark/numpy_financial_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'io',
    title: 'NumPy .npy and .npz Binary Serialization I/O',
    dartFile: 'benchmark/io_benchmark.dart',
    pythonFile: 'benchmark/numpy_io_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'strided_math',
    title: 'Non-Contiguous Strided Transposed Math (tan & exp)',
    dartFile: 'benchmark/strided_math_benchmark.dart',
    pythonFile: 'benchmark/numpy_strided_math_benchmark.py',
  ),
  BenchmarkSuiteSpec(
    id: 'strided_trig',
    title: 'Non-Contiguous Strided Transposed sin()',
    dartFile: 'benchmark/strided_trig_benchmark.dart',
    pythonFile: 'benchmark/numpy_strided_trig_benchmark.py',
  ),
];

final class BenchmarkPairRow {
  final String suiteId;
  final String suiteTitle;
  final String name;
  final double dartMeanUs;
  final double dartMedianUs;
  final double numpyMeanUs;
  final double numpyMedianUs;

  const BenchmarkPairRow({
    required this.suiteId,
    required this.suiteTitle,
    required this.name,
    required this.dartMeanUs,
    required this.dartMedianUs,
    required this.numpyMeanUs,
    required this.numpyMedianUs,
  });

  /// Ratio > 1.0 means `package:ndarray` is faster than NumPy.
  double get speedupMean => numpyMeanUs / dartMeanUs;

  /// Median speedup ratio (`numpyMedianUs / dartMedianUs`).
  double get speedupMedian => numpyMedianUs / dartMedianUs;
}

String _shortName(String fullName) {
  final idx = fullName.lastIndexOf(' / ');
  return idx >= 0 ? fullName.substring(idx + 3).trim() : fullName.trim();
}

String _formatSpeedup(double speedup) {
  if (speedup >= 1.0) {
    return '${speedup.toStringAsFixed(2)}x faster';
  } else {
    final inv = 1.0 / speedup;
    return '${inv.toStringAsFixed(2)}x slower (${speedup.toStringAsFixed(2)}x)';
  }
}

double _geometricMean(Iterable<double> values) {
  final list = values.where((v) => v > 0 && v.isFinite).toList();
  if (list.isEmpty) return 1.0;
  var logSum = 0.0;
  for (final v in list) {
    logSum += math.log(v);
  }
  return math.exp(logSum / list.length);
}

Future<void> main(List<String> args) async {
  final reuseDart = !args.contains('--run-dart');
  final reuseNumpy = args.contains('--reuse-numpy');
  String? suiteFilterArg;
  var markdownOut = 'benchmark/report/numpy_comparison.md';

  for (var i = 0; i < args.length; i++) {
    if (args[i] == '--suite' && i + 1 < args.length) {
      suiteFilterArg = args[++i];
    } else if (args[i].startsWith('--suite=')) {
      suiteFilterArg = args[i].substring('--suite='.length);
    } else if (args[i] == '--markdown' && i + 1 < args.length) {
      markdownOut = args[++i];
    }
  }

  final selectedIds = suiteFilterArg
      ?.split(',')
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toSet();

  final suites = selectedIds == null || selectedIds.isEmpty
      ? allSuites
      : allSuites.where((s) => selectedIds.contains(s.id)).toList();

  if (suites.isEmpty) {
    stderr.writeln(
      'No matching suites found for "$suiteFilterArg". Available IDs:\n'
      '  ${allSuites.map((s) => s.id).join(', ')}',
    );
    exit(2);
  }

  final singleThreadEnv = <String, String>{
    ...Platform.environment,
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
  };

  final allRows = <BenchmarkPairRow>[];
  String numpyVersion = 'unknown';

  print(
    '========================================================================================',
  );
  print(
    '  package:ndarray (Dart FFI/SIMD/OpenBLAS) vs NumPy (Python) Benchmark Comparison Runner',
  );
  print(
    '========================================================================================',
  );

  for (final suite in suites) {
    final dartJsonFile = File('benchmark/report/${suite.id}/results.json');
    final numpyJsonFile = File(
      'benchmark/report/${suite.id}/numpy_results.json',
    );

    if (!reuseDart || !dartJsonFile.existsSync()) {
      print('\n[Dart] Running ${suite.dartFile} ...');
      final res = await Process.run(Platform.resolvedExecutable, [
        'run',
        suite.dartFile,
      ], environment: singleThreadEnv);
      if (res.exitCode != 0) {
        stderr.writeln(
          'Dart benchmark ${suite.dartFile} failed:\n${res.stderr}',
        );
        exit(res.exitCode);
      }
    }

    if (!reuseNumpy || !numpyJsonFile.existsSync()) {
      print('[Python] Running ${suite.pythonFile} ...');
      final res = await Process.run('python3', [
        suite.pythonFile,
      ], environment: singleThreadEnv);
      if (res.exitCode != 0) {
        stderr.writeln(
          'Python benchmark ${suite.pythonFile} failed:\n${res.stderr}',
        );
        exit(res.exitCode);
      }
    }

    if (!dartJsonFile.existsSync() || !numpyJsonFile.existsSync()) {
      stderr.writeln('Missing JSON results for suite ${suite.id}; skipping.');
      continue;
    }

    final dartList =
        jsonDecode(dartJsonFile.readAsStringSync()) as List<dynamic>;
    final numpyMap =
        jsonDecode(numpyJsonFile.readAsStringSync()) as Map<String, dynamic>;
    numpyVersion = (numpyMap['numpy_version'] as String?) ?? numpyVersion;
    final numpyList = numpyMap['benchmarks'] as List<dynamic>;

    // Index Dart benchmarks by both full name and short name
    final dartByFull = <String, Map<String, dynamic>>{};
    final dartByShort = <String, Map<String, dynamic>>{};
    for (final item in dartList.cast<Map<String, dynamic>>()) {
      final fullName = item['name'] as String;
      dartByFull[fullName] = item;
      dartByShort[_shortName(fullName)] = item;
    }

    final shortCounts = <String, int>{};
    for (final npItem in numpyList.cast<Map<String, dynamic>>()) {
      final npFull = npItem['name'] as String;
      final npShort = (npItem['short_name'] as String?) ?? _shortName(npFull);
      shortCounts[npShort] = (shortCounts[npShort] ?? 0) + 1;
    }

    final suiteRows = <BenchmarkPairRow>[];
    for (final npItem in numpyList.cast<Map<String, dynamic>>()) {
      final npFull = npItem['name'] as String;
      final npShort = (npItem['short_name'] as String?) ?? _shortName(npFull);
      final dartItem = dartByFull[npFull] ?? dartByShort[npShort];
      if (dartItem == null) continue;

      final primary = dartItem['primary'] as Map<String, dynamic>;
      final dartMeanUs = (primary['mean'] as num).toDouble() / 1000.0;
      final dartMedianUs = (primary['median'] as num).toDouble() / 1000.0;
      final numpyMeanUs = (npItem['mean_us'] as num).toDouble();
      final numpyMedianUs = (npItem['median_us'] as num).toDouble();
      final displayName = (shortCounts[npShort] ?? 0) > 1 ? npFull : npShort;

      final row = BenchmarkPairRow(
        suiteId: suite.id,
        suiteTitle: suite.title,
        name: displayName,
        dartMeanUs: dartMeanUs,
        dartMedianUs: dartMedianUs,
        numpyMeanUs: numpyMeanUs,
        numpyMedianUs: numpyMedianUs,
      );
      suiteRows.add(row);
      allRows.add(row);
    }

    if (suiteRows.isNotEmpty) {
      final geo = _geometricMean(suiteRows.map((r) => r.speedupMean));
      print(
        '\n--- Suite: ${suite.title} (${suite.id}) | Geomean: ${_formatSpeedup(geo)} ---',
      );
      print(
        '${"Benchmark".padRight(52)} | ${"ndarray (us)".padLeft(12)} | ${"NumPy (us)".padLeft(12)} | Relative (ndarray vs NumPy)',
      );
      print('-' * 110);
      for (final r in suiteRows) {
        final label = r.name.length > 52 ? r.name.substring(0, 52) : r.name;
        print(
          '${label.padRight(52)} | ${r.dartMeanUs.toStringAsFixed(2).padLeft(12)} | ${r.numpyMeanUs.toStringAsFixed(2).padLeft(12)} | ${_formatSpeedup(r.speedupMean)}',
        );
      }
    }
  }

  if (allRows.isEmpty) {
    print('No matched benchmarks found.');
    return;
  }

  final overallGeoMean = _geometricMean(allRows.map((r) => r.speedupMean));
  final overallGeoMedian = _geometricMean(allRows.map((r) => r.speedupMedian));
  final fasterCount = allRows.where((r) => r.speedupMean >= 1.0).length;

  print(
    '\n========================================================================================',
  );
  print(
    '  OVERALL SUMMARY ACROSS ${allRows.length} PAIRED BENCHMARKS (${suites.length} SUITES)',
  );
  print(
    '  - ndarray faster or equal in: $fasterCount / ${allRows.length} benchmarks (${(fasterCount * 100.0 / allRows.length).toStringAsFixed(1)}%)',
  );
  print(
    '  - Geometric Mean Speedup (mean):   ${_formatSpeedup(overallGeoMean)}',
  );
  print(
    '  - Geometric Mean Speedup (median): ${_formatSpeedup(overallGeoMedian)}',
  );
  print(
    '========================================================================================',
  );

  // Generate Markdown & JSON reports
  final mdBuf = StringBuffer()
    ..writeln('# `package:ndarray` vs `NumPy` ($numpyVersion) Benchmark Report')
    ..writeln()
    ..writeln(
      'All benchmarks executed single-threaded (`setNumThreads(1)` / `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`).',
    )
    ..writeln()
    ..writeln('## Executive Summary')
    ..writeln()
    ..writeln(
      '- **Total Paired Benchmarks**: ${allRows.length} across ${suites.length} suites',
    )
    ..writeln(
      '- **`ndarray` Faster/Equal Count**: $fasterCount / ${allRows.length} (${(fasterCount * 100.0 / allRows.length).toStringAsFixed(1)}%)',
    )
    ..writeln(
      '- **Overall Geometric Mean (Mean Latency)**: **${_formatSpeedup(overallGeoMean)}**',
    )
    ..writeln(
      '- **Overall Geometric Mean (Median Latency)**: **${_formatSpeedup(overallGeoMedian)}**',
    )
    ..writeln()
    ..writeln('## Suite-Level Geometric Mean Summary')
    ..writeln()
    ..writeln(
      '| Suite ID | Suite Title | Benchmarks | Geomean (ndarray vs NumPy) |',
    )
    ..writeln('| :--- | :--- | :---: | :--- |');

  for (final suite in suites) {
    final sRows = allRows.where((r) => r.suiteId == suite.id).toList();
    if (sRows.isEmpty) continue;
    final geo = _geometricMean(sRows.map((r) => r.speedupMean));
    mdBuf.writeln(
      '| `${suite.id}` | ${suite.title} | ${sRows.length} | **${_formatSpeedup(geo)}** |',
    );
  }

  for (final suite in suites) {
    final sRows = allRows.where((r) => r.suiteId == suite.id).toList();
    if (sRows.isEmpty) continue;
    final geo = _geometricMean(sRows.map((r) => r.speedupMean));
    mdBuf
      ..writeln()
      ..writeln('### ${suite.title} (`${suite.id}`)')
      ..writeln()
      ..writeln('Suite geometric mean: **${_formatSpeedup(geo)}**')
      ..writeln()
      ..writeln(
        '| Benchmark | `ndarray` Mean (μs) | `NumPy` Mean (μs) | Speedup (`ndarray` vs `NumPy`) |',
      )
      ..writeln('| :--- | :---: | :---: | :--- |');
    for (final r in sRows) {
      mdBuf.writeln(
        '| `${r.name}` | ${r.dartMeanUs.toStringAsFixed(2)} | ${r.numpyMeanUs.toStringAsFixed(2)} | ${_formatSpeedup(r.speedupMean)} |',
      );
    }
  }

  final outFile = File(markdownOut);
  outFile.parent.createSync(recursive: true);
  outFile.writeAsStringSync(mdBuf.toString());

  final summaryJsonFile = File('benchmark/report/numpy_comparison.json');
  summaryJsonFile.writeAsStringSync(
    const JsonEncoder.withIndent('  ').convert({
      'numpy_version': numpyVersion,
      'total_benchmarks': allRows.length,
      'ndarray_faster_count': fasterCount,
      'overall_geomean_speedup_mean': overallGeoMean,
      'overall_geomean_speedup_median': overallGeoMedian,
      'benchmarks': [
        for (final r in allRows)
          {
            'suite_id': r.suiteId,
            'suite_title': r.suiteTitle,
            'name': r.name,
            'ndarray_mean_us': r.dartMeanUs,
            'ndarray_median_us': r.dartMedianUs,
            'numpy_mean_us': r.numpyMeanUs,
            'numpy_median_us': r.numpyMedianUs,
            'speedup_mean': r.speedupMean,
            'speedup_median': r.speedupMedian,
          },
      ],
    }),
  );

  print('\nWrote Markdown report to: $markdownOut');
  print('Wrote JSON summary to:    ${summaryJsonFile.path}');
}
