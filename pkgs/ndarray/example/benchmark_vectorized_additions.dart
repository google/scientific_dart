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

import 'dart:io';

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  const arraySize = 100000;
  const reportDir = 'benchmark/report/vectorized_additions';

  try {
    await NDArray.scope(() async {
      await criterion(
        'NDArray Float32 SIMD Additions Fast Path Benchmark',
        (c) {
          final a = linspace<DTypeTag>(
            1.0,
            100.0,
            arraySize,
            dtype: DType.float32,
          );
          final b = linspace<DTypeTag>(
            1.0,
            100.0,
            arraySize,
            dtype: DType.float32,
          );

          final viewA = a.reshape([arraySize ~/ 2, 2]).transposed;
          final viewB = b.reshape([arraySize ~/ 2, 2]).transposed;
          final outStrided = NDArray.create([2, arraySize ~/ 2], DType.float32);

          c.group('Float32 Addition Paths', () {
            c.bench(
              '1. Vectorized SIMD Additions Fast Path (Float32x4List)',
              () {
                NDArray.scope(() {
                  final res = add(a, b);
                  blackhole(res);
                });
              },
              throughput: Throughput.elements(arraySize),
            );

            c.bench(
              '2. Non-Contiguous Strided Additions Fallback (Pure loops)',
              () {
                add(viewA, viewB, out: outStrided);
              },
              throughput: Throughput.elements(arraySize),
            );
          });
        },
        config: CriterionConfig(
          generateHtmlReport: false,
          exportJson: false,
          reportDir: reportDir,
        ),
      );
    });
  } finally {
    final dir = Directory('benchmark/report');
    if (dir.existsSync()) {
      dir.deleteSync(recursive: true);
    }
  }
}
