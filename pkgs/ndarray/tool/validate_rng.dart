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

import 'dart:math' as math;

import 'package:ndarray/ndarray.dart';

void main() {
  print(
    '============================================================================',
  );
  print('            ndarray PRNG STATISTICAL QUALITY VALIDATION TOOL');
  print(
    '============================================================================',
  );

  NDArray.scope(() {
    runUniformTests();
    runNormalTests();
    runRandintTests();
  });

  print(
    '============================================================================',
  );
  print('            STATISTICAL QUALITY VALIDATION COMPLETE!');
  print(
    '============================================================================',
  );
}

void runUniformTests() {
  print('\n--- 1. CONTINUOUS UNIFORM DISTRIBUTION TESTS (Size: 100,000) ---');
  print(
    'Targeting Uniform [0.0, 1.0) - Theoretical Mean: 0.5000, Theoretical Var: 0.0833\n',
  );

  final size = 100000;

  // Standard: xoshiro256**
  final uStd = uniform([size], dtype: DType.float64, seed: 42);
  _evaluateUniform('xoshiro256** (Standard FFI)', uStd);

  // Secure: /dev/urandom
  final uSec = uniform([size], dtype: DType.float64, secure: true);
  _evaluateUniform('/dev/urandom (Secure FFI)', uSec);
}

void _evaluateUniform(String label, NDArray<double> arr) {
  final len = arr.data.length;
  final data = arr.data;

  // 1. Mean
  var sum = 0.0;
  for (var i = 0; i < len; i++) {
    sum += data[i];
  }
  final mean = sum / len;

  // 2. Variance
  var sumSqDiff = 0.0;
  for (var i = 0; i < len; i++) {
    final diff = data[i] - mean;
    sumSqDiff += diff * diff;
  }
  final variance = sumSqDiff / len;

  // 3. Chi-Square Goodness-of-Fit (10 Bins)
  final binCount = 10;
  final expectedCount = len / binCount;
  final bins = List<int>.filled(binCount, 0);
  for (var i = 0; i < len; i++) {
    var binIndex = (data[i] * binCount).floor();
    if (binIndex < 0) binIndex = 0;
    if (binIndex >= binCount) binIndex = binCount - 1;
    bins[binIndex]++;
  }

  var chiSq = 0.0;
  for (var i = 0; i < binCount; i++) {
    final diff = bins[i] - expectedCount;
    chiSq += (diff * diff) / expectedCount;
  }

  // 4. Lag-1 Autocorrelation
  var numAuto = 0.0;
  var denAuto = 0.0;
  for (var i = 0; i < len - 1; i++) {
    numAuto += (data[i] - mean) * (data[i + 1] - mean);
  }
  for (var i = 0; i < len; i++) {
    final diff = data[i] - mean;
    denAuto += diff * diff;
  }
  final lag1Auto = numAuto / denAuto;

  // Critical Chi-Square at 9 Degrees of Freedom, significance 0.05 is 16.919
  final chiSqCritical = 16.919;
  final uniformPass = chiSq <= chiSqCritical;

  print('[$label]');
  print(
    '  - Empirical Mean      : ${mean.toStringAsFixed(5)} (Target: 0.50000)',
  );
  print(
    '  - Empirical Variance  : ${variance.toStringAsFixed(5)} (Target: 0.08333)',
  );
  print(
    '  - Lag-1 Autocorrelation: ${lag1Auto.toStringAsFixed(5)} (Target: 0.00000)',
  );
  print(
    '  - Chi-Square Statistic: ${chiSq.toStringAsFixed(3)} (Critical threshold: $chiSqCritical)',
  );
  print(
    '  - Chi-Square Result   : ${uniformPass ? "PASS" : "FAIL"} (Uniformity verified)',
  );
  print('');
}

void runNormalTests() {
  print('--- 2. NORMAL (GAUSSIAN) DISTRIBUTION TESTS (Size: 100,000) ---');
  print(
    'Targeting Normal (0.0, 1.0) - Theoretical Mean: 0.0000, Theoretical SD: 1.0000\n',
  );

  final size = 100000;

  // Standard: xoshiro256**
  final nStd = normal(
    [size],
    loc: 0.0,
    scale: 1.0,
    dtype: DType.float64,
    seed: 42,
  );
  _evaluateNormal('xoshiro256** (Standard FFI)', nStd);

  // Secure: /dev/urandom
  final nSec = normal(
    [size],
    loc: 0.0,
    scale: 1.0,
    dtype: DType.float64,
    secure: true,
  );
  _evaluateNormal('/dev/urandom (Secure FFI)', nSec);
}

void _evaluateNormal(String label, NDArray<double> arr) {
  final len = arr.data.length;
  final data = arr.data;

  // 1. Mean
  var sum = 0.0;
  for (var i = 0; i < len; i++) {
    sum += data[i];
  }
  final mean = sum / len;

  // 2. Standard Deviation (SD)
  var sumSqDiff = 0.0;
  for (var i = 0; i < len; i++) {
    final diff = data[i] - mean;
    sumSqDiff += diff * diff;
  }
  final sd = math.sqrt(sumSqDiff / len);

  // 3. Skewness (Symmetric distribution should have skewness = 0.0)
  var sumCubedDiff = 0.0;
  for (var i = 0; i < len; i++) {
    final diff = data[i] - mean;
    sumCubedDiff += diff * diff * diff;
  }
  final skewness = (sumCubedDiff / len) / (sd * sd * sd);

  // 4. Excess Kurtosis (Normal distribution should have excess kurtosis = 0.0)
  var sumQuartDiff = 0.0;
  for (var i = 0; i < len; i++) {
    final diff = data[i] - mean;
    sumQuartDiff += diff * diff * diff * diff;
  }
  final excessKurtosis = ((sumQuartDiff / len) / (sd * sd * sd * sd)) - 3.0;

  print('[$label]');
  print(
    '  - Empirical Mean      : ${mean.toStringAsFixed(5)} (Target: 0.00000)',
  );
  print('  - Empirical Std Dev   : ${sd.toStringAsFixed(5)} (Target: 1.00000)');
  print(
    '  - Empirical Skewness  : ${skewness.toStringAsFixed(5)} (Target: 0.00000)',
  );
  print(
    '  - Excess Kurtosis     : ${excessKurtosis.toStringAsFixed(5)} (Target: 0.00000)',
  );
  print('');
}

void runRandintTests() {
  print('--- 3. DISCRETE UNIFORM (RANDINT) TESTS (Size: 100,000) ---');
  print(
    'Rolling a 6-Sided Die (randint from 1 to 7) - Expected Frequency: 16.67% each\n',
  );

  final size = 100000;

  // Standard: xoshiro256**
  final rStd = randint([size], low: 1, high: 7, dtype: DType.int64, seed: 42);
  _evaluateRandint('xoshiro256** (Standard FFI)', rStd);

  // Secure: /dev/urandom
  final rSec = randint(
    [size],
    low: 1,
    high: 7,
    dtype: DType.int64,
    secure: true,
  );
  _evaluateRandint('/dev/urandom (Secure FFI)', rSec);
}

void _evaluateRandint(String label, NDArray<int> arr) {
  final len = arr.data.length;
  final data = arr.data;

  final frequencies = <int, int>{};
  for (var i = 1; i <= 6; i++) {
    frequencies[i] = 0;
  }

  for (var i = 0; i < len; i++) {
    final val = data[i];
    frequencies[val] = (frequencies[val] ?? 0) + 1;
  }

  print('[$label]');
  for (var i = 1; i <= 6; i++) {
    final count = frequencies[i] ?? 0;
    final percentage = (count / len) * 100.0;
    print(
      '  - Face $i frequency: ${percentage.toStringAsFixed(2)}% ($count rolls)',
    );
  }
  print('');
}
