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

import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Random Probability Distributions Examples ===\n');
  runNormalDistributionExample();
  runExponentialDistributionExample();
  runPoissonDistributionExample();
  runBinomialDistributionExample();
  runUniformDistributionExample();
  runRandintDistributionExample();
  runSeededReproducibilityExample();
  runChoiceShufflePermutationExample();
  runAudioImageRandintExample();
}

void runChoiceShufflePermutationExample() {
  print('\n--- Choice, Shuffle, & Permutation ---');
  NDArray.scope(() {
    final a = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);
    print('Original array a: ${a.toList()}');

    // Sample 3 items without replacement
    final sampled = choice(a, size: [3], replace: false, seed: 42);
    print('choice(a, size: [3], replace: false): ${sampled.toList()}');

    // Shuffling in-place
    final toShuffle = a.copy();
    shuffle(toShuffle, seed: 100);
    print('shuffle(a) in-place: ${toShuffle.toList()}');

    // Permutation (returns permuted copy)
    final perm = permutation(a, seed: 200);
    print('permutation(a): ${perm.toList()}');
    print('Original array still intact: ${a.toList()}');
  });
}

void runAudioImageRandintExample() {
  print('\n--- Media Types Expansion (uint8 Image & int16 Audio) ---');
  NDArray.scope(() {
    // uint8 Image Pixel Sampling
    final pixels = randint([4, 4], low: 0, high: 256, dtype: DType.uint8);
    print('4x4 uint8 Image Pixel Grid:\n${pixels.toList()}');
    print('Image DType: ${pixels.dtype}');

    // int16 Audio Sample Sampling
    final audio = randint([10], low: -32768, high: 32767, dtype: DType.int16);
    print('10-element int16 Audio Samples: ${audio.toList()}');
    print('Audio DType: ${audio.dtype}');
  });
}

void runNormalDistributionExample() {
  print('--- Normal (Gaussian) Distribution ---');
  // Draw a 2x3 normal array with mean=0.0 and stddev=1.0
  final a = normal([2, 3], loc: 0.0, scale: 1.0, dtype: DType.float64);
  print('Normal shape: ${a.shape}');
  print('Data: ${a.toList()}');

  // Draw with a shifted center and scale
  final b = normal([5], loc: 100.0, scale: 15.0);
  print('Shifted normal (loc=100, scale=15): ${b.toList()}');
}

void runExponentialDistributionExample() {
  print('\n--- Exponential Distribution ---');
  // Draw exponential distribution with scale parameter (beta) = 2.0
  final a = exponential([5], scale: 2.0);
  print('Exponential data (scale=2.0): ${a.toList()}');
}

void runPoissonDistributionExample() {
  print('\n--- Poisson Distribution ---');
  // Small lambda using Knuth exact algorithm
  final smallLam = poisson([5], lam: 4.0, dtype: DType.int64);
  print('Poisson data (small lam=4.0, Knuth): ${smallLam.toList()}');

  // Large lambda using Gaussian approximation to avoid performance stalls
  final largeLam = poisson([5], lam: 50.0, dtype: DType.int64);
  print('Poisson data (large lam=50.0, Gaussian approx): ${largeLam.toList()}');
}

void runBinomialDistributionExample() {
  print('\n--- Binomial Distribution (Bernoulli Trials) ---');
  // Small n: 20 trials with success probability 0.5
  final smallN = binomial([5], n: 20, p: 0.5, dtype: DType.int32);
  print('Binomial data (n=20, p=0.5, Bernoulli trials): ${smallN.toList()}');

  // Large n: 1000 trials with p=0.3, triggers fast normal approximation
  final largeN = binomial([5], n: 1000, p: 0.3, dtype: DType.int64);
  print('Binomial data (n=1000, p=0.3, Normal approx): ${largeN.toList()}');
}

void runSeededReproducibilityExample() {
  print('\n--- Seeded Reproducibility (Seeding for Science) ---');
  // Passing the same seed object guarantees exactly identical random draws!
  final draw1 = normal([3], loc: 10.0, scale: 2.0, seed: 42);
  final draw2 = normal([3], loc: 10.0, scale: 2.0, seed: 42);

  print('Draw 1 (Seed 42): ${draw1.toList()}');
  print('Draw 2 (Seed 42): ${draw2.toList()}');
  print(
    'Do draws match perfectly? ${draw1[0] == draw2[0] && draw1[1] == draw2[1] ? "YES" : "NO"}',
  );
}

void runUniformDistributionExample() {
  print('\n--- Uniform Distribution ---');
  // Draw a 2x3 uniform array with values between 0.0 and 1.0
  final a = uniform([2, 3], dtype: DType.float64);
  print('Uniform shape: ${a.shape}');
  print('Data: ${a.toList()}');
}

void runRandintDistributionExample() {
  print('\n--- Uniform Integer (randint) Distribution ---');
  // Draw a 2x3 uniform integer array with values in [0, 10)
  final a = randint([2, 3], low: 0, high: 10, dtype: DType.int64);
  print('Randint shape: ${a.shape}');
  print('Data: ${a.toList()}');
}
