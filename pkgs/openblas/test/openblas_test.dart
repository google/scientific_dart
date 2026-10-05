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
import 'dart:math' as math;

import 'package:ffi/ffi.dart';
import 'package:openblas/openblas.dart';
import 'package:test/test.dart';

void main() {
  group('OpenBLAS Tests', () {
    test('Get number of threads', () {
      final threads = openblas_get_num_threads();
      print('OpenBLAS threads: $threads');
      expect(threads, greaterThan(0));
    });

    test('Get config', () {
      final configPtr = openblas_get_config();
      expect(configPtr, isNot(ffi.nullptr));
      final configStr = configPtr.cast<Utf8>().toDartString();
      print('OpenBLAS Config: $configStr');
      expect(configStr, isNotEmpty);
    });

    test('Vector dot product (cblas_sdot)', () {
      final n = 3;
      final x = calloc<ffi.Float>(n);
      final y = calloc<ffi.Float>(n);

      x[0] = 1.0;
      x[1] = 2.0;
      x[2] = 3.0;
      y[0] = 4.0;
      y[1] = 5.0;
      y[2] = 6.0;

      final result = cblas_sdot(n, x, 1, y, 1);
      print('sdot result: $result');

      expect(result, closeTo(32.0, 0.0001));

      calloc.free(x);
      calloc.free(y);
    });

    test('Vector addition (cblas_saxpy)', () {
      final n = 3;
      final alpha = 2.0;
      final x = calloc<ffi.Float>(n);
      final y = calloc<ffi.Float>(n);

      x[0] = 1.0;
      x[1] = 2.0;
      x[2] = 3.0;
      y[0] = 4.0;
      y[1] = 5.0;
      y[2] = 6.0;

      cblas_saxpy(n, alpha, x, 1, y, 1);

      print('saxpy result y[0]: ${y[0]}');
      print('saxpy result y[1]: ${y[1]}');
      print('saxpy result y[2]: ${y[2]}');

      expect(y[0], closeTo(6.0, 0.0001));
      expect(y[1], closeTo(9.0, 0.0001));
      expect(y[2], closeTo(12.0, 0.0001));

      calloc.free(x);
      calloc.free(y);
    });
  });

  group('Double-precision and complex CBLAS bindings', () {
    const cblasRowMajor = 101;
    const cblasNoTrans = 111;
    const cblasTrans = 112;

    test(
      'Level-1 CBLAS (cblas_ddot, cblas_dnrm2, cblas_daxpy, cblas_dscal)',
      () {
        const n = 4;
        final x = calloc<ffi.Double>(n);
        final y = calloc<ffi.Double>(n);
        try {
          for (var i = 0; i < n; i++) {
            x[i] = (i + 1).toDouble(); // [1, 2, 3, 4]
            y[i] = 2.0 * (i + 1); // [2, 4, 6, 8]
          }
          expect(cblas_ddot(n, x, 1, y, 1), closeTo(60.0, 1e-12));
          expect(cblas_dnrm2(n, x, 1), closeTo(math.sqrt(30.0), 1e-12));
          cblas_daxpy(n, 3.0, x, 1, y, 1); // y = 3*x + y = 5*x
          for (var i = 0; i < n; i++) {
            expect(y[i], closeTo(5.0 * (i + 1), 1e-12));
          }
          cblas_dscal(n, 0.5, x, 1);
          for (var i = 0; i < n; i++) {
            expect(x[i], closeTo(0.5 * (i + 1), 1e-12));
          }
        } finally {
          calloc.free(x);
          calloc.free(y);
        }
      },
    );

    test('Level-2 and Level-3 CBLAS (cblas_dgemv, cblas_dgemm)', () {
      final a = calloc<ffi.Double>(4);
      final b = calloc<ffi.Double>(4);
      final c = calloc<ffi.Double>(4);
      final x = calloc<ffi.Double>(2);
      final y = calloc<ffi.Double>(2);
      try {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        b[0] = 5.0;
        b[1] = 6.0;
        b[2] = 7.0;
        b[3] = 8.0;
        cblas_dgemm(
          cblasRowMajor,
          cblasNoTrans,
          cblasNoTrans,
          2,
          2,
          2,
          1.0,
          a,
          2,
          b,
          2,
          0.0,
          c,
          2,
        );
        expect(c[0], closeTo(19.0, 1e-12));
        expect(c[1], closeTo(22.0, 1e-12));
        expect(c[2], closeTo(43.0, 1e-12));
        expect(c[3], closeTo(50.0, 1e-12));

        x[0] = 2.0;
        x[1] = 3.0;
        cblas_dgemv(
          cblasRowMajor,
          cblasNoTrans,
          2,
          2,
          1.0,
          a,
          2,
          x,
          1,
          0.0,
          y,
          1,
        );
        expect(y[0], closeTo(8.0, 1e-12));
        expect(y[1], closeTo(18.0, 1e-12));
      } finally {
        calloc.free(a);
        calloc.free(b);
        calloc.free(c);
        calloc.free(x);
        calloc.free(y);
      }
    });

    test('Complex CBLAS (cblas_zgemm)', () {
      // 1x1 complex multiplication: (2 + 3i) * (4 - i) = 11 + 10i
      final a = calloc<ffi.Double>(2);
      final b = calloc<ffi.Double>(2);
      final c = calloc<ffi.Double>(2);
      final alpha = calloc<ffi.Double>(2);
      final beta = calloc<ffi.Double>(2);
      try {
        a[0] = 2.0;
        a[1] = 3.0;
        b[0] = 4.0;
        b[1] = -1.0;
        alpha[0] = 1.0;
        alpha[1] = 0.0;
        beta[0] = 0.0;
        beta[1] = 0.0;
        cblas_zgemm(
          cblasRowMajor,
          cblasNoTrans,
          cblasNoTrans,
          1,
          1,
          1,
          alpha,
          a,
          1,
          b,
          1,
          beta,
          c,
          1,
        );
        expect(c[0], closeTo(11.0, 1e-12));
        expect(c[1], closeTo(10.0, 1e-12));
      } finally {
        calloc.free(a);
        calloc.free(b);
        calloc.free(c);
        calloc.free(alpha);
        calloc.free(beta);
      }
    });

    test('1x1 degenerate cblas_dgemm applies alpha and beta scaling', () {
      final a = calloc<ffi.Double>(1)..value = -3.5;
      final b = calloc<ffi.Double>(1)..value = 4.0;
      final c = calloc<ffi.Double>(1)..value = 10.0;
      try {
        // c = 2.0 * (-3.5 * 4.0) + 0.5 * 10.0 = -28.0 + 5.0 = -23.0
        cblas_dgemm(
          cblasRowMajor,
          cblasNoTrans,
          cblasNoTrans,
          1,
          1,
          1,
          2.0,
          a,
          1,
          b,
          1,
          0.5,
          c,
          1,
        );
        expect(c.value, closeTo(-23.0, 1e-12));
      } finally {
        calloc.free(a);
        calloc.free(b);
        calloc.free(c);
      }
    });

    test('Non-unit stride (incx > 1, incy > 1) in cblas_ddot', () {
      final x = calloc<ffi.Double>(6);
      final y = calloc<ffi.Double>(6);
      try {
        for (var i = 0; i < 6; i++) {
          x[i] = (i + 1).toDouble();
          y[i] = 10.0 * (i + 1);
        }
        // Elements at indices 0, 2, 4: x=[1, 3, 5], y=[10, 30, 50] -> dot = 10 + 90 + 250 = 350
        final dot = cblas_ddot(3, x, 2, y, 2);
        expect(dot, closeTo(350.0, 1e-12));
      } finally {
        calloc.free(x);
        calloc.free(y);
      }
    });

    test('Transposed operand flag in cblas_dgemm', () {
      final a = calloc<ffi.Double>(4);
      final c = calloc<ffi.Double>(4);
      try {
        // A = [[1, 2], [3, 4]], C = A^T * A = [[10, 14], [14, 20]]
        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        cblas_dgemm(
          cblasRowMajor,
          cblasTrans,
          cblasNoTrans,
          2,
          2,
          2,
          1.0,
          a,
          2,
          a,
          2,
          0.0,
          c,
          2,
        );
        expect(c[0], closeTo(10.0, 1e-12));
        expect(c[1], closeTo(14.0, 1e-12));
        expect(c[2], closeTo(14.0, 1e-12));
        expect(c[3], closeTo(20.0, 1e-12));
      } finally {
        calloc.free(a);
        calloc.free(c);
      }
    });
  });
}
