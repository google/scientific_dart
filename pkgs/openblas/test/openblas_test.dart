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
}
