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

void main() async {
  const size = 4; // Simulating 4D shape/strides

  await criterion(
    'Allocation vs Buffer Reuse Overhead',
    (c) {
      final buffer = malloc<ffi.Int>(size);

      c.group('FFI Native Buffer Allocation', () {
        c.bench('Malloc + Free per iteration', () {
          final ptr = malloc<ffi.Int>(size);
          ptr[0] = 1;
          ptr[1] = 2;
          ptr[2] = 3;
          ptr[3] = 4;
          blackhole(ptr[0]);
          malloc.free(ptr);
        });

        c.bench('Buffer Reuse (preallocated)', () {
          final ptr = buffer;
          ptr[0] = 1;
          ptr[1] = 2;
          ptr[2] = 3;
          ptr[3] = 4;
          blackhole(ptr[0]);
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/allocation',
    ),
  );
}
