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

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:gpuarray/gpuarray.dart';
import 'package:test/test.dart';

void main() {
  group('Safetensors Serialization (F11)', () {
    late Directory temporaryDirectory;

    setUp(() {
      temporaryDirectory = Directory.systemTemp.createTempSync(
        'gpuarray_safetensors_test_',
      );
    });

    tearDown(() {
      if (temporaryDirectory.existsSync()) {
        temporaryDirectory.deleteSync(recursive: true);
      }
    });

    test(
      'round-trips contiguous Float64 and Float32 tensors via file and bytes',
      () {
        final path = '${temporaryDirectory.path}/model.safetensors';
        final weights = GpuArray.fromList(
          <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final bias = GpuArray.fromList(
          <double>[-1.5, 0.5, 2.5],
          [3],
          DType.float32,
        );

        try {
          saveSafetensorsFile(
            path,
            <String, GpuArray>{'weights': weights, 'bias': bias},
            metadata: <String, String>{'format': 'pt'},
          );

          final loaded = loadSafetensorsFile(path);
          try {
            expect(loaded.keys, containsAll(<String>['weights', 'bias']));
            final loadedWeights = loaded['weights']! as GpuArray<Float64>;
            final loadedBias = loaded['bias']! as GpuArray<Float32>;

            expect(loadedWeights.shape, equals(<int>[2, 3]));
            expect(loadedWeights.dtype, equals(DType.float64));
            expect(
              loadedWeights.toList(),
              equals(<double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            );

            expect(loadedBias.shape, equals(<int>[3]));
            expect(loadedBias.dtype, equals(DType.float32));
            expect(loadedBias.toList(), equals(<double>[-1.5, 0.5, 2.5]));
          } finally {
            for (final tensor in loaded.values) {
              tensor.dispose();
            }
          }
        } finally {
          weights.dispose();
          bias.dispose();
        }
      },
    );

    test('correctly serializes non-contiguous transposed views', () {
      final matrix = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        DType.float64,
      );
      final transposed = matrix.transpose();

      try {
        expect(transposed.isContiguous, isFalse);
        final bytes = saveSafetensors(<String, GpuArray>{'t': transposed});

        final loaded = loadSafetensors(bytes);
        try {
          final loadedTensor = loaded['t']! as GpuArray<Float64>;
          expect(loadedTensor.shape, equals(<int>[3, 2]));
          expect(
            loadedTensor.toList(),
            equals(<double>[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
          );
        } finally {
          for (final tensor in loaded.values) {
            tensor.dispose();
          }
        }
      } finally {
        transposed.dispose();
        matrix.dispose();
      }
    });

    test('round-trips integer and boolean dtypes', () {
      final ints = GpuArray.fromList(
        <int>[10, 20, 30, 40],
        [2, 2],
        DType.int32,
      );
      final flags = GpuArray.fromList(
        <bool>[true, false, true],
        [3],
        DType.boolean,
      );

      try {
        final bytes = saveSafetensors(<String, GpuArray>{
          'ints': ints,
          'flags': flags,
        });

        final loaded = loadSafetensors(bytes);
        try {
          final loadedInts = loaded['ints']! as GpuArray<Int32>;
          final loadedFlags = loaded['flags']! as GpuArray<Bool>;
          expect(loadedInts.toList(), equals(<int>[10, 20, 30, 40]));
          expect(loadedFlags.toList(), equals(<bool>[true, false, true]));
        } finally {
          for (final tensor in loaded.values) {
            tensor.dispose();
          }
        }
      } finally {
        ints.dispose();
        flags.dispose();
      }
    });

    test('throws FormatException on truncated or malformed payloads', () {
      expect(
        () => loadSafetensors(Uint8List.fromList(<int>[1, 2, 3])),
        throwsA(isA<FormatException>()),
      );

      final badHeaderBytes = ByteData(16)..setUint64(0, 100, Endian.little);
      expect(
        () => loadSafetensors(badHeaderBytes.buffer.asUint8List()),
        throwsA(isA<FormatException>()),
      );

      final jsonBytes = utf8.encode('["not_a_json_object"]');
      final badJsonBuffer = BytesBuilder();
      final lengthData = ByteData(8)
        ..setUint64(0, jsonBytes.length, Endian.little);
      badJsonBuffer.add(lengthData.buffer.asUint8List());
      badJsonBuffer.add(jsonBytes);
      expect(
        () => loadSafetensors(badJsonBuffer.toBytes()),
        throwsA(isA<FormatException>()),
      );
    });

    test('throws FormatException on mismatched data_offsets byte length', () {
      final headerJson = jsonEncode(<String, Object>{
        'w': <String, Object>{
          'dtype': 'F64',
          'shape': <int>[2, 2],
          'data_offsets': <int>[0, 16], // Expects 32 bytes for 2x2 F64
        },
      });
      final headerBytes = utf8.encode(headerJson);
      final builder = BytesBuilder();
      final lengthData = ByteData(8)
        ..setUint64(0, headerBytes.length, Endian.little);
      builder.add(lengthData.buffer.asUint8List());
      builder.add(headerBytes);
      builder.add(Uint8List(32));

      expect(
        () => loadSafetensors(builder.toBytes()),
        throwsA(isA<FormatException>()),
      );
    });

    test('validates programmer preconditions on saveSafetensors', () {
      final tensor = GpuArray.fromList(<double>[1.0, 2.0], [2], DType.float64);
      try {
        expect(
          () => saveSafetensors(<String, GpuArray>{'__metadata__': tensor}),
          throwsArgumentError,
        );
      } finally {
        tensor.dispose();
      }
      expect(
        () => saveSafetensors(<String, GpuArray>{'disposed': tensor}),
        throwsStateError,
      );
    });
  });
}
