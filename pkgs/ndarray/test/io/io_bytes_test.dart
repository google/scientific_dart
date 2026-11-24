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

import 'package:archive/archive.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

const String _wasmTempRoot = String.fromEnvironment(
  'NDARRAY_TEST_TMPDIR',
  defaultValue: '/tmp',
);

void main() {
  Directory? tempDir;
  late String tempDirPath;

  setUpAll(() {
    if (_isWasm) {
      tempDirPath =
          '$_wasmTempRoot/ndarray_io_bytes_test_${DateTime.now().microsecondsSinceEpoch}';
    } else {
      tempDir = Directory.systemTemp.createTempSync('ndarray_io_bytes_test_');
      tempDirPath = tempDir!.path;
    }
  });

  tearDownAll(() {
    if (!_isWasm && tempDir != null && tempDir!.existsSync()) {
      tempDir!.deleteSync(recursive: true);
    }
  });

  tearDown(NDArray.checkNoLeaks);

  group('saveBytes / loadBytes (.npy in-memory)', () {
    test('round-trips all 15 DTypes cleanly', () {
      NDArray.scope(() {
        final cases = <NDArray<DTypeTag>>[
          NDArray.fromList([1.5, -2.5, 3.25, 0.0], [2, 2], DType.float64),
          NDArray.fromList([1.25, -0.5, 4.0], [3], DType.float32),
          NDArray.fromList([1.0, -2.0, 0.5], [3], DType.float16),
          NDArray.fromList([1.0, -2.0, 3.5], [3], DType.bfloat16),
          NDArray.fromList([-10000000000, 0, 42], [3], DType.int64),
          NDArray.fromList([-100, 0, 200], [3], DType.int32),
          NDArray.fromList([-32000, 12345], [2], DType.int16),
          NDArray.fromList([-128, 0, 127], [3], DType.int8),
          NDArray.fromList([0, 1, 9007199254740991], [3], DType.uint64),
          NDArray.fromList([0, 4000000000], [2], DType.uint32),
          NDArray.fromList([0, 65535], [2], DType.uint16),
          NDArray.fromList([0, 42, 255], [3], DType.uint8),
          NDArray.fromList(
            [const Complex(1.5, -2.5), const Complex(0.0, 4.0)],
            [2],
            DType.complex128,
          ),
          NDArray.fromList(
            [const Complex(0.5, 1.5), const Complex(-3.0, 2.0)],
            [2],
            DType.complex64,
          ),
          NDArray.fromList([true, false, true, false], [2, 2], DType.boolean),
        ];

        for (final original in cases) {
          final bytes = saveBytes(original);
          expect(bytes.lengthInBytes, greaterThanOrEqualTo(64));
          expect(bytes.sublist(0, 6), [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]);

          final restored = loadBytes(bytes);
          expect(restored.dtype, original.dtype);
          expect(restored.shape, original.shape);
          expect(restored.toBytes(), equals(original.toBytes()));
        }
      });
    });

    test('handles 0-D scalar, empty shapes, and non-contiguous views', () {
      NDArray.scope(() {
        final scalar = NDArray.fromList([42.5], [], DType.float64);
        final scalarOut = loadBytes(saveBytes(scalar));
        expect(scalarOut.shape, isEmpty);
        expect(scalarOut.toList(), [42.5]);

        final empty = NDArray.zeros([2, 0, 3], DType.int32);
        final emptyOut = loadBytes(saveBytes(empty));
        expect(emptyOut.shape, [2, 0, 3]);
        expect(emptyOut.dtype, DType.int32);
        expect(emptyOut.size, 0);

        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final transposed = mat.transpose();
        expect(transposed.isContiguous, isFalse);
        final transposedOut = loadBytes(saveBytes(transposed));
        expect(transposedOut.shape, [3, 2]);
        expect(transposedOut.toList(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
      });
    });

    test('respects non-zero Uint8List.sublistView offsetInBytes', () {
      NDArray.scope(() {
        final original = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final raw = saveBytes(original);
        final padded = Uint8List(raw.length + 11);
        padded.setRange(7, 7 + raw.length, raw);
        final subview = Uint8List.sublistView(padded, 7, 7 + raw.length);
        expect(subview.offsetInBytes, 7);

        final restored = loadBytes(subview);
        expect(restored.shape, [2, 2]);
        expect(restored.toList(), [10.0, 20.0, 30.0, 40.0]);
      });
    });

    test('loads Fortran-order and v2.0 .npy byte streams accurately', () {
      NDArray.scope(() {
        // Construct a Fortran-order 2x3 float64 NPY v2.0 buffer.
        const headerBody =
            "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
        final paddedLen = ((12 + headerBody.length + 1) + 63) ~/ 64 * 64 - 12;
        final headerStr =
            '$headerBody${' ' * (paddedLen - headerBody.length - 1)}\n';
        final headerUnits = ascii.encode(headerStr);
        final payload = Float64List.fromList([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        final payloadBytes = payload.buffer.asUint8List();

        final npy = Uint8List(12 + headerUnits.length + payloadBytes.length);
        npy.setAll(0, [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x02, 0x00]);
        ByteData.sublistView(
          npy,
          8,
          12,
        ).setUint32(0, headerUnits.length, Endian.little);
        npy.setRange(12, 12 + headerUnits.length, headerUnits);
        npy.setRange(12 + headerUnits.length, npy.length, payloadBytes);

        final loaded = loadBytes(npy);
        expect(loaded.shape, [2, 3]);
        expect(loaded.strides, [1, 2]);
        expect(loaded.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      });
    });

    test('matches save() / load() filesystem representation bit-for-bit', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final path = '$tempDirPath/parity.npy';
        save(path, a);
        final fromMem = saveBytes(a);
        if (!_isWasm) {
          expect(fromMem, equals(File(path).readAsBytesSync()));
        }
        expect(loadBytes(fromMem).toList(), equals(load(path).toList()));
      });
    });

    test('validates disposed arrays and malformed NPY inputs', () {
      final disposed = NDArray.zeros([2], DType.float64)..dispose();
      expect(() => saveBytes(disposed), throwsStateError);

      // Too short / bad magic / bad version / truncated header / truncated body
      expect(() => loadBytes(Uint8List(3)), throwsFormatException);
      expect(
        () => loadBytes(Uint8List.fromList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        throwsFormatException,
      );
      expect(
        () => loadBytes(
          Uint8List.fromList([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x04, 0x00]),
        ),
        throwsFormatException,
      );
      // v2 header length exceeding 1 MiB limit
      final hugeHeader = Uint8List(12)
        ..setAll(0, [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x02, 0x00]);
      ByteData.sublistView(
        hugeHeader,
        8,
        12,
      ).setUint32(0, 2 * 1024 * 1024, Endian.little);
      expect(() => loadBytes(hugeHeader), throwsFormatException);

      // Header claims 64 bytes of header, buffer ends early
      final shortHeader = Uint8List(16)
        ..setAll(0, [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00, 54, 0]);
      expect(() => loadBytes(shortHeader), throwsFormatException);

      // Valid header, truncated payload
      NDArray.scope(() {
        final valid = saveBytes(NDArray.zeros([4], DType.float64));
        expect(
          () => loadBytes(valid.sublist(0, valid.length - 4)),
          throwsFormatException,
        );
      });
    });
  });

  group('savezBytes / loadzBytes (.npz in-memory)', () {
    test('round-trips uncompressed and compressed multi-array archives', () {
      for (final compressed in [false, true]) {
        NDArray.scope(() {
          final weights = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final transposed = weights.transpose();
          final labels = NDArray.fromList([7, 8, 9], [3], DType.int32);
          final mask = NDArray.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          final empty = NDArray.zeros([0, 2], DType.float32);

          final npzBytes = savezBytes({
            'weights': weights,
            'transposed': transposed,
            'labels': labels,
            'mask': mask,
            'empty': empty,
          }, compressed: compressed);

          expect(npzBytes.lengthInBytes, greaterThan(22));
          // Standard ZIP local file header signature "PK\x03\x04"
          expect(npzBytes.sublist(0, 4), [0x50, 0x4B, 0x03, 0x04]);

          // Verify interoperability with package:archive ZipDecoder
          final decodedZip = ZipDecoder().decodeBytes(npzBytes);
          expect(
            decodedZip.files.map((f) => f.name).toSet(),
            equals({
              'weights.npy',
              'transposed.npy',
              'labels.npy',
              'mask.npy',
              'empty.npy',
            }),
          );

          // Load back via loadzBytes (including from an unaligned subview)
          final padded = Uint8List(npzBytes.length + 9);
          padded.setRange(5, 5 + npzBytes.length, npzBytes);
          final subview = Uint8List.sublistView(padded, 5, 5 + npzBytes.length);
          final loaded = loadzBytes(subview);
          expect(
            loaded.keys.toSet(),
            equals({'weights', 'transposed', 'labels', 'mask', 'empty'}),
          );
          expect(loaded['weights']!.shape, [2, 3]);
          expect(loaded['weights']!.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
          expect(loaded['transposed']!.shape, [3, 2]);
          expect(loaded['transposed']!.toList(), [
            1.0,
            4.0,
            2.0,
            5.0,
            3.0,
            6.0,
          ]);
          expect(loaded['labels']!.dtype, DType.int32);
          expect(loaded['labels']!.toList(), [7, 8, 9]);
          expect(loaded['mask']!.dtype, DType.boolean);
          expect(loaded['mask']!.toList(), [true, false, true]);
          expect(loaded['empty']!.shape, [0, 2]);
          expect(loaded['empty']!.size, 0);
        });
      }
    });

    test('interoperates with savez() / loadz() and skips non-.npy entries', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.5, 2.5, 3.5], [3], DType.float32);
        final path = '$tempDirPath/parity.npz';
        savez(path, {'a': a}, compressed: true);
        if (!_isWasm) {
          final loadedFromFileBytes = loadzBytes(File(path).readAsBytesSync());
          expect(loadedFromFileBytes['a']!.toList(), [1.5, 2.5, 3.5]);
        }

        // Archive containing a directory, a text readme, and a valid .npy entry.
        final npy = saveBytes(a);
        final archive = Archive()
          ..addFile(ArchiveFile.noCompress('meta/', 0, Uint8List(0)))
          ..addFile(ArchiveFile('README.txt', 5, ascii.encode('hello')))
          ..addFile(ArchiveFile('a.npy', npy.length, npy));
        final zipBytes = Uint8List.fromList(ZipEncoder().encode(archive)!);
        final loaded = loadzBytes(zipBytes);
        expect(loaded.keys.toList(), ['a']);
        expect(loaded['a']!.toList(), [1.5, 2.5, 3.5]);
      });
    });

    test('validates empty map, disposed entries, and corrupt archives', () {
      expect(() => savezBytes({}), throwsArgumentError);

      final disposed = NDArray.zeros([2], DType.float64)..dispose();
      expect(() => savezBytes({'x': disposed}), throwsStateError);

      expect(() => loadzBytes(Uint8List(0)), throwsFormatException);
      expect(() => loadzBytes(Uint8List(10)), throwsFormatException);
      expect(() => loadzBytes(Uint8List(64)), throwsFormatException);
    });
  });
}
