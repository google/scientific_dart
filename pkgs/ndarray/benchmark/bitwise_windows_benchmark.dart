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

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);
  const size = 100000;

  await criterion(
    'NDArray Bitwise, Windows & Special Functions Benchmark Suite',
    (c) {
      c.group('1. DSP Windowing Functions (100k points)', () {
        c.bench('hanning(100k)', () {
          final res = hanning<DTypeTag>(size);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('hamming(100k)', () {
          final res = hamming<DTypeTag>(size);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Special Mathematical Functions (100k elements)', () {
        final floatVec = linspace<DTypeTag>(
          0.0,
          10.0,
          size,
          dtype: DType.float64,
        );

        c.bench('i0(x) (Bessel I0) [100k]', () {
          final res = i0((floatVec as NDArray<AnySpec>));
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('sinc(x) (Normalized Sinc) [100k]', () {
          final res = sinc((floatVec as NDArray<AnySpec>));
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('3. Bitwise Integer Operations (100k elements)', () {
        final intA = NDArray.fromList(List.generate(size, (i) => i * 13), [
          size,
        ], DType.int32);
        final intB = NDArray.fromList(List.generate(size, (i) => i * 7 + 1), [
          size,
        ], DType.int32);

        c.bench('bitwiseAnd(a, b) [100k Int32]', () {
          final res = bitwiseAnd(intA, intB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('bitwiseOr(a, b) [100k Int32]', () {
          final res = bitwiseOr(intA, intB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('bitwiseXor(a, b) [100k Int32]', () {
          final res = bitwiseXor(intA, intB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('invert(a) [100k Int32]', () {
          final res = invert(intA);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        final shiftAmt = NDArray.fromList(List.generate(size, (i) => (i % 8)), [
          size,
        ], DType.int32);

        c.bench('leftShift(a, shift) [100k Int32]', () {
          final res = leftShift(intA, shiftAmt);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('rightShift(a, shift) [100k Int32]', () {
          final res = rightShift(intA, shiftAmt);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/bitwise_windows',
    ),
  );
}
