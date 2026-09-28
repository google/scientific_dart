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

import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/random.dart' as rng;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Pseudo-Random Number Generation (gpuarray.random)', () {
    test(
      'Philox4x32 uniform random sampling and 16-bit split arithmetic (rand, uniform)',
      () {
        ResourceScope.scope(() {
          // Test with large 64-bit seed that would previously cause 64-bit integer overflow
          rng.seed(0x7FFFFFFFFFFFFFFF);
          final u = rng.rand([1000]);
          expect(u.shape, equals([1000]));
          expect(u.dtype, equals(DType.float64));

          final uList = u.toList().cast<double>();
          var sum = 0.0;
          for (final val in uList) {
            expect(val, greaterThanOrEqualTo(0.0));
            expect(val, lessThan(1.0));
            sum += val;
          }
          final mean = sum / uList.length;
          expect(mean, closeTo(0.5, 0.05)); // Uniform(0, 1) mean is 0.5

          // Custom uniform range [10.0, 20.0)
          final customU = rng.uniform(low: 10.0, high: 20.0, shape: [500]);
          final cList = customU.toList().cast<double>();
          var cSum = 0.0;
          for (final val in cList) {
            expect(val, greaterThanOrEqualTo(10.0));
            expect(val, lessThan(20.0));
            cSum += val;
          }
          expect(cSum / cList.length, closeTo(15.0, 0.5));
        });
      },
    );

    test('PRNG reproducibility across seed invocations', () {
      ResourceScope.scope(() {
        rng.seed(987654321);
        final r1 = rng.rand([20]).toList().cast<double>();

        rng.seed(987654321);
        final r2 = rng.rand([20]).toList().cast<double>();

        expect(r1, equals(r2));
      });
    });

    test(
      'Gaussian normal sampling backed by Philox Box-Muller (randn, normal)',
      () {
        ResourceScope.scope(() {
          rng.seed(123);
          final g = rng.randn([2000]);
          expect(g.shape, equals([2000]));

          final gList = g.toList().cast<double>();
          var sum = 0.0;
          var sumSq = 0.0;
          for (final val in gList) {
            sum += val;
            sumSq += val * val;
          }
          final mean = sum / gList.length;
          final variance = (sumSq / gList.length) - (mean * mean);
          expect(mean, closeTo(0.0, 0.1)); // Sample mean of N(0, 1) near 0
          expect(
            variance,
            closeTo(1.0, 0.15),
          ); // Sample variance of N(0, 1) near 1

          // Custom normal N(10.0, 4.0) -> loc=10, scale=2.0
          final norm = rng.normal(loc: 10.0, scale: 2.0, shape: [2000]);
          final normList = norm.toList().cast<double>();
          var nSum = 0.0;
          for (final val in normList) {
            nSum += val;
          }
          expect(nSum / normList.length, closeTo(10.0, 0.2));
        });
      },
    );

    test(
      'Random integers and exponential backed by Philox (randint, exponential)',
      () {
        ResourceScope.scope(() {
          rng.seed(42);
          final ints = rng.randint(5, 15, [100]);
          expect(ints.shape, equals([100]));
          expect(ints.dtype, equals(DType.int64));
          final intList = ints.toList().cast<int>();
          for (final v in intList) {
            expect(v, greaterThanOrEqualTo(5));
            expect(v, lessThan(15));
          }

          // Int32 dtype
          final int32s = rng.randint(-10, 10, [50], DType.int32);
          expect(int32s.dtype, equals(DType.int32));
          final i32List = int32s.toList().cast<int>();
          for (final v in i32List) {
            expect(v, greaterThanOrEqualTo(-10));
            expect(v, lessThan(10));
          }

          // Invalid randint bounds
          expect(() => rng.randint(10, 5), throwsArgumentError);
          expect(() => rng.randint(10, 10), throwsArgumentError);

          // Exponential distribution
          final exp = rng.exponential(scale: 2.0, shape: [2000]);
          final expList = exp.toList().cast<double>();
          var eSum = 0.0;
          for (final v in expList) {
            expect(v, greaterThanOrEqualTo(0.0));
            eSum += v;
          }
          expect(eSum / expList.length, closeTo(2.0, 0.15));
        });
      },
    );

    test('Random choice with and without replacement, and weighted p', () {
      ResourceScope.scope(() {
        rng.seed(42);
        final arr = GpuArray.fromList(
          [10.0, 20.0, 30.0, 40.0, 50.0],
          [5],
          DType.float64,
        );

        // Uniform with replacement
        final chReplace = rng.choice(arr, size: 10, replace: true);
        expect(chReplace.shape, equals([10]));
        for (final item in chReplace.toList().cast<double>()) {
          expect(arr.toList().cast<double>(), contains(item));
        }

        // Uniform without replacement
        final chNoReplace = rng.choice(arr, size: 4, replace: false);
        expect(chNoReplace.shape, equals([4]));
        final sampleList = chNoReplace.toList().cast<double>();
        // All elements must be unique because replace = false
        expect(sampleList.toSet().length, equals(4));
        for (final item in sampleList) {
          expect(arr.toList().cast<double>(), contains(item));
        }

        // Weighted with replacement (heavy bias towards 50.0)
        final p = [0.0, 0.0, 0.0, 0.0, 1.0];
        final weighted = rng.choice(arr, size: 20, replace: true, p: p);
        final wList = weighted.toList().cast<double>();
        for (final v in wList) {
          expect(v, equals(50.0));
        }

        // Weighted without replacement
        final pWeights = [0.1, 0.2, 0.3, 0.4, 0.0];
        final wNoRep = rng.choice(arr, size: 3, replace: false, p: pWeights);
        expect(wNoRep.shape, equals([3]));
        final wNoRepList = wNoRep.toList().cast<double>();
        expect(wNoRepList.toSet().length, equals(3));
        expect(wNoRepList, isNot(contains(50.0))); // Weight for 50.0 is 0.0

        // Error checking for choice
        expect(
          () => rng.choice(arr, size: 10, replace: false),
          throwsArgumentError,
        );
        expect(
          () => rng.choice(arr, size: 3, p: [0.1, 0.2]), // Length mismatch
          throwsArgumentError,
        );
        expect(
          () => rng.choice(
            arr,
            size: 3,
            p: [-0.1, 0.5, 0.2, 0.2, 0.2],
          ), // Negative prob
          throwsArgumentError,
        );
        expect(
          () =>
              rng.choice(arr, size: 3, p: [0.0, 0.0, 0.0, 0.0, 0.0]), // Sum = 0
          throwsArgumentError,
        );

        // Insufficient positive probabilities when replace = false
        expect(
          () => rng.choice(
            arr,
            size: 3,
            replace: false,
            p: [0.5, 0.5, 0.0, 0.0, 0.0],
          ),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains(
                'Cannot take 3 samples without replacement when only 2 items have positive probability.',
              ),
            ),
          ),
        );
      });
    });

    test('Random permutation and shuffle', () {
      ResourceScope.scope(() {
        rng.seed(1234);
        final perm = rng.permutation(5) as GpuArray;
        expect(perm.shape, equals([5]));
        expect(perm.toList().cast<int>().toSet().length, equals(5));

        final orig = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        rng.shuffle(orig);
        expect(orig.toList().cast<double>().toSet().length, equals(5));
      });
    });

    test('Shuffle on sliced subarray with non-zero offset', () {
      ResourceScope.scope(() {
        rng.seed(42);
        final base = GpuArray.fromList(
          [100.0, 200.0, 1.0, 2.0, 3.0, 4.0, 300.0, 400.0],
          [8],
          DType.float64,
        );

        final sub = base.slice([Slice(2, 6)]);
        expect(sub.offsetElements, equals(2));
        expect(sub.shape, equals([4]));

        rng.shuffle(sub);

        final baseList = base.toList().cast<double>();
        // Unmodified prefix and suffix
        expect(baseList[0], equals(100.0));
        expect(baseList[1], equals(200.0));
        expect(baseList[6], equals(300.0));
        expect(baseList[7], equals(400.0));

        // Subarray is shuffled in place within its slice
        final subList = baseList.sublist(2, 6);
        expect(subList.toSet(), equals({1.0, 2.0, 3.0, 4.0}));
      });
    });
  });
}
