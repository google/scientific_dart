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
import 'package:ndarray/ndarray.dart';
import 'package:openblas/openblas.dart';

void main() async {
  setNumThreads(1);

  final sizes = [10, 100, 1000, 10000, 100000, 1000000];

  await criterion(
    'NDArray Addition & Vectorization Benchmarks',
    (c) {
      // 1. Float64 Addition (Dart out-parameter vs FFI cblas_daxpy)
      c.group('Float64 Addition (Pre-allocated Output)', () {
        for (final size in sizes) {
          final a = NDArray<double>.ones([size], DType.float64);
          final b = NDArray<double>.ones([size], DType.float64);
          final out = NDArray<double>.create([size], DType.float64);

          c.bench('Dart add(out: outBuf) [$size]', () {
            add(a, b, out: out);
          }, throughput: Throughput.elements(size));

          c.bench('FFI cblas_daxpy [$size]', () {
            cblas_daxpy(
              size,
              1.0,
              b.pointer.cast<ffi.Double>(),
              1,
              out.pointer.cast<ffi.Double>(),
              1,
            );
          }, throughput: Throughput.elements(size));
        }
      });

      c.group('Float64 Addition (With New Allocation)', () {
        c.benchWith<void, int>('Dart add(a, b)', sizes, (size) {
          NDArray.scope(() {
            final a = NDArray<double>.ones([size], DType.float64);
            final b = NDArray<double>.ones([size], DType.float64);
            final r = add(a, b);
            blackhole(r);
          });
        }, throughput: (size) => Throughput.elements(size));
      });

      // 2. Float32 Addition Non-Contiguous Strided (Scalar vs SIMD vs FFI)
      c.group('Float32 Addition (Non-Contiguous Strided Step=2)', () {
        final stridedSizes = [100, 1000, 10000, 100000];
        for (final size in stridedSizes) {
          final sizeHalf = size ~/ 2;
          final a = NDArray<double>.ones([size], DType.float32);
          final b = NDArray<double>.ones([size], DType.float32);
          final aView = NDArray<double>.view(
            a,
            shape: [sizeHalf],
            strides: [2],
          );
          final bView = NDArray<double>.view(
            b,
            shape: [sizeHalf],
            strides: [2],
          );
          final outView = NDArray<double>.create([sizeHalf], DType.float32);

          final aContig = NDArray<double>.ones([sizeHalf], DType.float32);
          final bContig = NDArray<double>.ones([sizeHalf], DType.float32);
          final outContig = NDArray<double>.create([sizeHalf], DType.float32);

          c.bench('Scalar Strided add() [$sizeHalf]', () {
            add(aView, bView, out: outView);
          }, throughput: Throughput.elements(sizeHalf));

          c.bench('Contiguous SIMD add() [$sizeHalf]', () {
            add(aContig, bContig, out: outContig);
          }, throughput: Throughput.elements(sizeHalf));

          c.bench('Contiguous FFI cblas_saxpy [$sizeHalf]', () {
            cblas_saxpy(
              sizeHalf,
              1.0,
              bContig.pointer.cast<ffi.Float>(),
              1,
              outContig.pointer.cast<ffi.Float>(),
              1,
            );
          }, throughput: Throughput.elements(sizeHalf));
        }
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/addition',
    ),
  );
}
