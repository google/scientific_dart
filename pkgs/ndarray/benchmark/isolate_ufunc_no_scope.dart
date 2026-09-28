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

import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_bindings.dart';

void main() {
  final size = 100000;
  final x = NDArray.ones([size], DType.float64);
  final out = NDArray.create([size], DType.float64);

  // Warm up
  for (var i = 0; i < 100; i++) {
    v_sin_double(x.pointer.cast(), out.pointer.cast(), size, ffi.nullptr);
    sin(x, out: out);
  }

  // 1. Benchmark raw FFI call only
  final swFFI = Stopwatch()..start();
  for (var i = 0; i < 1000; i++) {
    v_sin_double(x.pointer.cast(), out.pointer.cast(), size, ffi.nullptr);
  }
  swFFI.stop();
  print('Raw FFI v_sin_double (1000 runs): ${swFFI.elapsedMilliseconds} ms');

  // 2. Benchmark full wrapper sin()
  final swWrapper = Stopwatch()..start();
  for (var i = 0; i < 1000; i++) {
    sin(x, out: out);
  }
  swWrapper.stop();
  print('Wrapper sin() (1000 runs): ${swWrapper.elapsedMilliseconds} ms');

  x.dispose();
  out.dispose();
}
