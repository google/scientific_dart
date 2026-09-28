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

import 'package:gpuarray/gpuarray.dart';
import 'package:test/test.dart';

void main() {
  group('Philox4x32Engine & RandomState (F10)', () {
    test('Philox4x32Engine is deterministic and supports reset/nextBlock', () {
      final engineA = Philox4x32Engine(seed: 42);
      final engineB = Philox4x32Engine(seed: 42);

      expect(engineA.nextBlock(), equals(engineB.nextBlock()));
      expect(engineA.counter, equals(4));

      engineA.reset(seed: 42, counter: 16);
      engineB.reset(seed: 42, counter: 16);
      expect(engineA.nextBlock(), equals(engineB.nextBlock()));

      expect(
        () => Philox4x32Engine.philox4x32TenRounds(<int>[1, 2], <int>[0, 0]),
        throwsArgumentError,
      );
    });

    test('RandomState uniform respects bounds and out: parameter', () {
      final rngA = RandomState(12345);
      final rngB = RandomState(12345);

      final sampleA = rngA.uniform(shape: <int>[16], low: -2.0, high: 3.0);
      final outB = GpuArray.zeros([16], DType.float64);
      try {
        final sampleB = rngB.uniform(
          shape: <int>[16],
          low: -2.0,
          high: 3.0,
          out: outB,
        );
        expect(identical(sampleB, outB), isTrue);
        final listA = sampleA.toList().cast<double>();
        final listB = sampleB.toList().cast<double>();
        expect(listA, equals(listB));
        for (final element in listA) {
          expect(element, greaterThanOrEqualTo(-2.0));
          expect(element, lessThan(3.0));
        }
      } finally {
        sampleA.dispose();
        outB.dispose();
      }
    });

    test('RandomState normal and standardNormal support out: parameter', () {
      final rng = RandomState(99);
      final normalOut = GpuArray.zeros([32], DType.float64);
      final stdOut = GpuArray.zeros([32], DType.float64);
      try {
        final normalSample = rng.normal(
          shape: <int>[32],
          loc: 1.0,
          scale: 0.5,
          out: normalOut,
        );
        expect(identical(normalSample, normalOut), isTrue);

        final stdSample = rng.standardNormal(shape: <int>[32], out: stdOut);
        expect(identical(stdSample, stdOut), isTrue);
        for (final element in stdSample.toList().cast<double>()) {
          expect(element.isFinite, isTrue);
        }
      } finally {
        normalOut.dispose();
        stdOut.dispose();
      }
    });

    test('RandomState randint, exponential, and choice', () {
      final rng = RandomState(2026);
      final integers = rng.randint(5, 10, <int>[20]);
      final waitingTimes = rng.exponential(scale: 2.0, shape: <int>[20]);
      final population = GpuArray.fromList(
        <int>[10, 20, 30, 40],
        [4],
        DType.int64,
      );
      try {
        for (final element in integers.toList().cast<int>()) {
          expect(element, greaterThanOrEqualTo(5));
          expect(element, lessThan(10));
        }
        for (final element in waitingTimes.toList().cast<double>()) {
          expect(element, greaterThanOrEqualTo(0.0));
        }

        final choices = rng.choice(population, shape: <int>[3], replace: false);
        try {
          expect(choices.shape, equals(<int>[3]));
          final selected = choices.toList().cast<int>();
          expect(selected.toSet().length, equals(3));
          for (final item in selected) {
            expect(<int>[10, 20, 30, 40], contains(item));
          }
        } finally {
          choices.dispose();
        }
      } finally {
        integers.dispose();
        waitingTimes.dispose();
        population.dispose();
      }
    });

    test('RandomState permutation and shuffle support out: and in-place', () {
      final rng = RandomState(777);
      final outPerm = GpuArray.zeros([8], DType.int64);
      try {
        final perm = rng.permutation<Int64>(8, out: outPerm);
        expect(identical(perm, outPerm), isTrue);
        final sorted = perm.toList().cast<int>().toList()..sort();
        expect(sorted, equals(<int>[0, 1, 2, 3, 4, 5, 6, 7]));

        rng.shuffle(outPerm);
        final shuffledSorted = outPerm.toList().cast<int>().toList()..sort();
        expect(shuffledSorted, equals(<int>[0, 1, 2, 3, 4, 5, 6, 7]));
      } finally {
        outPerm.dispose();
      }
    });

    test('RandomState validates arguments', () {
      final rng = RandomState(1);
      expect(
        () => rng.uniform(shape: <int>[4], low: 5.0, high: 2.0),
        throwsArgumentError,
      );
      expect(
        () => rng.normal(shape: <int>[4], scale: 0.0),
        throwsArgumentError,
      );
      expect(() => rng.randint(3, 3, <int>[4]), throwsArgumentError);
      expect(
        () => rng.exponential(shape: <int>[4], scale: -1.0),
        throwsArgumentError,
      );
      expect(() => rng.permutation<Int64>(-1), throwsArgumentError);
    });
  });
}
