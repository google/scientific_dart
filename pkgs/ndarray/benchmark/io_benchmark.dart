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
  setNumThreads(1);
  final tempDir = Directory.systemTemp.createTempSync('ndarray_io_bench_');
  final npyPath = '${tempDir.path}/array_1m.npy';
  final npzPath = '${tempDir.path}/archive.npz';
  final npzCompressedPath = '${tempDir.path}/archive_compressed.npz';

  const elementCount = 1000000; // 1M float64 = 8MB
  const byteSize = elementCount * 8;

  try {
    await criterion(
      'NDArray IO Serialization & Deserialization Benchmark Suite',
      (c) {
        final rawArray = linspace(
          0.0,
          100.0,
          elementCount,
          dtype: DType.float64,
        );

        c.group('1. Binary NumPy (.npy) File IO', () {
          c.bench('save() 8MB float64 array to .npy', () {
            save(npyPath, rawArray);
          }, throughput: Throughput.bytes(byteSize));

          // Ensure file exists for load benchmark
          save(npyPath, rawArray);

          c.bench('load() 8MB float64 array from .npy', () {
            final loaded = load(npyPath);
            blackhole(loaded);
            loaded.dispose();
          }, throughput: Throughput.bytes(byteSize));
        });

        c.group('2. NumPy Zip Archive (.npz) IO', () {
          final halfArray = linspace(
            0.0,
            50.0,
            elementCount ~/ 2,
            dtype: DType.float64,
          );
          final multiArrays = {'arr_a': rawArray, 'arr_b': halfArray};

          c.bench('savez() 12MB multi-array archive (.npz)', () {
            savez(npzPath, multiArrays);
          }, throughput: Throughput.bytes(byteSize + (byteSize ~/ 2)));

          savez(npzPath, multiArrays);

          c.bench('loadz() 12MB multi-array archive (.npz)', () {
            final loadedMap = loadz(npzPath);
            for (final arr in loadedMap.values) {
              blackhole(arr);
              arr.dispose();
            }
          }, throughput: Throughput.bytes(byteSize + (byteSize ~/ 2)));

          c.bench('savez_compressed() Deflate 12MB (.npz)', () {
            savez(npzCompressedPath, multiArrays, compressed: true);
          }, throughput: Throughput.bytes(byteSize + (byteSize ~/ 2)));

          savez(npzCompressedPath, multiArrays, compressed: true);

          c.bench('loadz() compressed Deflate 12MB (.npz)', () {
            final loadedMap = loadz(npzCompressedPath);
            for (final arr in loadedMap.values) {
              blackhole(arr);
              arr.dispose();
            }
          }, throughput: Throughput.bytes(byteSize + (byteSize ~/ 2)));
        });
      },
      config: CriterionConfig(
        generateHtmlReport: true,
        exportJson: true,
        reportDir: 'benchmark/report/io',
      ),
    );
  } finally {
    if (tempDir.existsSync()) {
      tempDir.deleteSync(recursive: true);
    }
  }
}
