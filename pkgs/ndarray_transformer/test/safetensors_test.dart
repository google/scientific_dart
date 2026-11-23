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

import 'package:ndarray/ndarray.dart';
import 'package:ndarray_transformer/ndarray_transformer.dart';
import 'package:test/test.dart';

void main() {
  group('SafetensorsFile', () {
    test('round-trips multiple dtypes, metadata, and strided views', () {
      NDArray.scope(() {
        final f32 = NDArray<Float32>.fromTypedData(
          Float32List.fromList([1.5, -2.25, 3.0, 4.5, 5.125, -6.0]),
          [2, 3],
          DType.float32,
        );
        final f64 = NDArray<Float64>.fromTypedData(
          Float64List.fromList([0.1, 0.2, 0.3]),
          [3],
          DType.float64,
        );
        final i64 = NDArray<Int64>.fromTypedData(
          Int64List.fromList([-10, 0, 42, 9999999999]),
          [2, 2],
          DType.int64,
        );
        final f16 = f32.asType(DType.float16);
        final bf16 = f32.asType(DType.bfloat16);
        final mask = NDArray<Boolean>.fromList(
          [true, false, true],
          [3],
          DType.boolean,
        );
        // Non-contiguous transposed view should serialize cleanly in C-order.
        final f32T = f32.transposed;

        final bytes = SafetensorsFile.encode(
          {
            'w_f32': f32,
            'w_f32_t': f32T,
            'w_f64': f64,
            'ids': i64,
            'w_f16': f16,
            'w_bf16': bf16,
            'mask': mask,
          },
          metadata: {'format': 'pt', 'model': 'unit-test'},
        );

        final parsed = SafetensorsFile.fromBytes(bytes);
        expect(parsed.metadata, {'format': 'pt', 'model': 'unit-test'});
        expect(parsed.contains('w_f32'), isTrue);
        expect(parsed.contains('missing'), isFalse);

        final gotF32 = parsed.loadFloat32('w_f32');
        expect(gotF32.shape, [2, 3]);
        expect(gotF32.getCellFlat(0), closeTo(1.5, 1e-6));
        expect(gotF32.getCellFlat(5), closeTo(-6.0, 1e-6));

        final gotF32T = parsed.loadFloat32('w_f32_t');
        expect(gotF32T.shape, [3, 2]);
        expect(gotF32T.getCell([0, 1]), closeTo(4.5, 1e-6));
        expect(gotF32T.getCell([2, 0]), closeTo(3.0, 1e-6));

        // F16, BF16, and F64 auto-cast cleanly through loadFloat32.
        final gotFromF16 = parsed.loadFloat32('w_f16');
        final gotFromBf16 = parsed.loadFloat32('w_bf16');
        final gotFromF64 = parsed.loadFloat32('w_f64');
        expect(gotFromF16.getCellFlat(1), closeTo(-2.25, 1e-2));
        expect(gotFromBf16.getCellFlat(1), closeTo(-2.25, 5e-2));
        expect(gotFromF64.getCellFlat(2), closeTo(0.3, 1e-6));

        final gotIds = parsed.loadTensor('ids') as NDArray<Int64>;
        expect(gotIds.shape, [2, 2]);
        expect(gotIds.getCellFlat(3), 9999999999);

        final gotMask = parsed.loadTensor('mask') as NDArray<Boolean>;
        expect(gotMask.getCellFlat(0), isTrue);
        expect(gotMask.getCellFlat(1), isFalse);
        expect(gotMask.getCellFlat(2), isTrue);

        // Non-floating tensor rejected by loadFloat32.
        expect(() => parsed.loadFloat32('ids'), throwsFormatException);
      });
    });

    test('lazy on-disk SafetensorsFile.open reads individual tensors', () {
      final tmp = Directory.systemTemp.createTempSync('st_test_');
      try {
        final file = File('${tmp.path}/weights.safetensors');
        NDArray.scope(() {
          final a = NDArray<Float32>.fromTypedData(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float32,
          );
          final b = NDArray<Float32>.fromTypedData(
            Float32List.fromList([10.0, 20.0]),
            [2],
            DType.float32,
          );
          SafetensorsFile.writeToFile(file, {'a': a, 'b': b});
        });

        final reader = SafetensorsFile.open(file);
        try {
          NDArray.scope(() {
            final b = reader.loadFloat32('b');
            expect(b.shape, [2]);
            expect(b.getCellFlat(0), 10.0);
            expect(b.getCellFlat(1), 20.0);
          });
        } finally {
          reader.close();
        }
        expect(reader.isClosed, isTrue);
        expect(() => reader.loadFloat32('a'), throwsStateError);
      } finally {
        tmp.deleteSync(recursive: true);
      }
    });

    test('rejects malformed headers, out-of-bounds, and overlaps', () {
      expect(
        () => SafetensorsFile.fromBytes(Uint8List(4)),
        throwsFormatException,
      );

      // Header length exceeds payload.
      final hugeHeader = Uint8List(16);
      ByteData.sublistView(hugeHeader).setUint64(0, 9999, Endian.little);
      expect(
        () => SafetensorsFile.fromBytes(hugeHeader),
        throwsFormatException,
      );

      // Helper to build raw safetensors with arbitrary header JSON + data size.
      Uint8List rawFrame(Map<String, Object?> header, int dataBytes) {
        final h = utf8.encode(jsonEncode(header));
        final out = Uint8List(8 + h.length + dataBytes);
        ByteData.sublistView(out, 0, 8).setUint64(0, h.length, Endian.little);
        out.setRange(8, 8 + h.length, h);
        return out;
      }

      // Byte length mismatch for shape/dtype.
      expect(
        () => SafetensorsFile.fromBytes(
          rawFrame({
            'x': {
              'dtype': 'F32',
              'shape': [2, 2],
              'data_offsets': [0, 8],
            },
          }, 16),
        ),
        throwsFormatException,
      );

      // Overlapping intervals.
      expect(
        () => SafetensorsFile.fromBytes(
          rawFrame({
            'a': {
              'dtype': 'F32',
              'shape': [2],
              'data_offsets': [0, 8],
            },
            'b': {
              'dtype': 'F32',
              'shape': [2],
              'data_offsets': [4, 12],
            },
          }, 12),
        ),
        throwsFormatException,
      );

      // Out-of-bounds end offset.
      expect(
        () => SafetensorsFile.fromBytes(
          rawFrame({
            'a': {
              'dtype': 'F32',
              'shape': [2],
              'data_offsets': [0, 8],
            },
          }, 4),
        ),
        throwsFormatException,
      );
    });
  });
}
