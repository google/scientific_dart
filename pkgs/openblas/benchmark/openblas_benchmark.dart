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

import 'dart:ffi' as ffi;
import 'package:criterion/criterion.dart';
import 'package:ffi/ffi.dart';
import 'package:openblas/openblas.dart';

void main() async {
  await criterion(
    'Native OpenBLAS C FFI Bindings Benchmarks',
    (c) {
      c.group('BLAS Level 1: DAXPY (y <- a*x + y)', () {
        final sizes = [100, 1000, 10000, 100000];
        for (final size in sizes) {
          final x = malloc<ffi.Double>(size);
          final y = malloc<ffi.Double>(size);
          for (var i = 0; i < size; i++) {
            x[i] = 1.0;
            y[i] = 2.0;
          }

          c.bench('cblas_daxpy [$size]', () {
            cblas_daxpy(size, 1.5, x, 1, y, 1);
          }, throughput: Throughput.elements(size));
        }
      });

      c.group('BLAS Level 3: DGEMM (C <- alpha*A*B + beta*C)', () {
        final dims = [32, 64, 128, 256];
        for (final n in dims) {
          final a = malloc<ffi.Double>(n * n);
          final b = malloc<ffi.Double>(n * n);
          final outC = malloc<ffi.Double>(n * n);

          for (var i = 0; i < n * n; i++) {
            a[i] = 0.1;
            b[i] = 0.2;
            outC[i] = 0.0;
          }

          c.bench(
            'cblas_dgemm [${n}x$n]',
            () {
              cblas_dgemm(
                101, // CblasRowMajor
                111, // CblasNoTrans
                111, // CblasNoTrans
                n,
                n,
                n,
                1.0,
                a,
                n,
                b,
                n,
                0.0,
                outC,
                n,
              );
              blackhole(outC[0]);
            },
            throughput: Throughput.elements(n * n * n), // FLOP scaling ~ 2*N^3
          );
        }
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/openblas',
    ),
  );
}
