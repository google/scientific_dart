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
import 'dart:typed_data';
import 'dart:io';
import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/safetensors.dart' as st;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray SafeTensors Serialization (gpuarray.safetensors)', () {
    test('Roundtrip in-memory SafeTensors serialization', () {
      ResourceScope.scope(() {
        final w = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float32,
        );
        final b = GpuArray.fromList([0.1, 0.2], [2], DType.float64);
        final mask = GpuArray.fromList(
          [true, false, true, true],
          [4],
          DType.boolean,
        );

        final tensorDict = {
          'model.weight': w,
          'model.bias': b,
          'model.mask': mask,
        };

        final bytes = st.saveSafetensors(
          tensorDict,
          metadata: {'format': 'pt'},
        );
        expect(bytes.isNotEmpty, isTrue);

        final loadedDict = st.loadSafetensors(bytes);
        expect(
          loadedDict.keys,
          containsAll(['model.weight', 'model.bias', 'model.mask']),
        );

        expect(loadedDict['model.weight']!.shape, equals([2, 3]));
        expect(loadedDict['model.weight']!.dtype, equals(DType.float32));
        expect(
          loadedDict['model.weight']!.toList().cast<double>(),
          equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );

        expect(loadedDict['model.bias']!.shape, equals([2]));
        expect(loadedDict['model.bias']!.dtype, equals(DType.float64));
        expect(
          loadedDict['model.bias']!.toList().cast<double>(),
          equals([0.1, 0.2]),
        );

        expect(loadedDict['model.mask']!.shape, equals([4]));
        expect(loadedDict['model.mask']!.dtype, equals(DType.boolean));
        expect(
          loadedDict['model.mask']!.toList().cast<bool>(),
          equals([true, false, true, true]),
        );
      });
    });

    test('Roundtrip file-based SafeTensors save and load', () {
      ResourceScope.scope(() {
        final tempDir = Directory.systemTemp.createTempSync('safetensors_test');
        final filePath = '${tempDir.path}/test_model.safetensors';

        try {
          final emb = GpuArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
          st.saveSafetensorsFile(filePath, {'embeddings': emb});

          final loaded = st.loadSafetensorsFile(filePath);
          expect(loaded.containsKey('embeddings'), isTrue);
          expect(loaded['embeddings']!.shape, equals([2, 2]));
          expect(loaded['embeddings']!.dtype, equals(DType.int32));
          expect(
            loaded['embeddings']!.toList().cast<int>(),
            equals([10, 20, 30, 40]),
          );
        } finally {
          tempDir.deleteSync(recursive: true);
        }
      });
    });
    test('SafeTensors validation on corrupt or invalid payloads', () {
      ResourceScope.scope(() {
        // Buffer smaller than 8 bytes
        expect(
          () => st.loadSafetensors(Uint8List.fromList([1, 2, 3])),
          throwsArgumentError,
        );

        // Header length exceeding buffer length
        final headerTooLarge = Uint8List(16);
        ByteData.sublistView(headerTooLarge).setUint64(0, 1000, Endian.little);
        expect(() => st.loadSafetensors(headerTooLarge), throwsArgumentError);

        // Valid header with invalid offsets (offsets end beyond buffer length)
        final badOffsetsDict = {
          'bad_tensor': {
            'dtype': 'F32',
            'shape': [2],
            'data_offsets': [0, 8],
          },
        };
        final headerJson = jsonEncode(badOffsetsDict);
        final headerBytes = utf8.encode(headerJson);
        final payload = Uint8List(
          8 + headerBytes.length + 4,
        ); // Only 4 bytes of data, needs 8
        ByteData.sublistView(
          payload,
        ).setUint64(0, headerBytes.length, Endian.little);
        payload.setRange(8, 8 + headerBytes.length, headerBytes);
        expect(() => st.loadSafetensors(payload), throwsFormatException);
      });
    });
  });
}
