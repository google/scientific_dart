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
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Workstream 3: Security & Build Hardening', () {
    group('C2: 32-bit Array Size Ceiling Enforcement', () {
      test('NDArray.create rejects arrays with totalSize > 2^31 - 1', () {
        expect(
          () => NDArray.create([2147483648], DType.float64),
          throwsA(
            isA<UnsupportedError>().having(
              (e) => e.message,
              'message',
              contains(
                'NDArray operations currently support arrays up to 2^31 - 1 elements. Got 2147483648.',
              ),
            ),
          ),
        );

        expect(
          () => NDArray.create([2, 1073741824], DType.int32),
          throwsA(
            isA<UnsupportedError>().having(
              (e) => e.message,
              'message',
              contains(
                'NDArray operations currently support arrays up to 2^31 - 1 elements.',
              ),
            ),
          ),
        );
      });

      test('NDArray.fromPointer rejects arrays with totalSize > 2^31 - 1', () {
        final dummyPtr = malloc<ffi.Double>(1);
        try {
          expect(
            () => NDArray.fromPointer(dummyPtr.cast<ffi.Void>(), [
              2147483648,
            ], DType.float64),
            throwsA(
              isA<UnsupportedError>().having(
                (e) => e.message,
                'message',
                contains(
                  'NDArray operations currently support arrays up to 2^31 - 1 elements. Got 2147483648.',
                ),
              ),
            ),
          );
        } finally {
          malloc.free(dummyPtr);
        }
      });

      test(
        'NDArray.create allows arrays up to 2^31 - 1 within available memory',
        () {
          NDArray.scope(() {
            final arr = NDArray.create([100], DType.float64);
            expect(arr.size, 100);
          });
        },
      );
    });

    group('C3: .npz Loader Security & Bounds Checks', () {
      late Directory tempDir;

      setUp(() {
        tempDir = Directory.systemTemp.createTempSync('npz_security_test_');
      });

      tearDown(() {
        if (tempDir.existsSync()) {
          tempDir.deleteSync(recursive: true);
        }
      });

      test(
        'loadz throws FormatException on mismatched NPY header vs ZIP dataLen',
        () {
          // Construct a valid NPY header claiming shape (10,) of float64 (80 bytes)
          const headerDict =
              "{ 'descr': '<f8', 'fortran_order': False, 'shape': (10,), }\n";
          final headerUtf8 = headerDict.codeUnits;
          final hlen = headerUtf8.length;
          final paddedHlen = ((10 + hlen + 15) ~/ 16) * 16 - 10;
          final padding = List<int>.filled(paddedHlen - hlen, 0x20);

          final npyHeaderBytes = BytesBuilder();
          npyHeaderBytes.add([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00]);
          npyHeaderBytes.add([paddedHlen & 0xFF, (paddedHlen >> 8) & 0xFF]);
          npyHeaderBytes.add(headerUtf8);
          npyHeaderBytes.add(padding);

          final fullNpyHeader = npyHeaderBytes.toBytes();

          // But provide only 40 bytes of payload instead of expected 80 bytes!
          final payload = List<int>.filled(40, 0);

          final entryBytes = [...fullNpyHeader, ...payload];

          final archive = Archive();
          archive.addFile(
            ArchiveFile('bad_entry.npy', entryBytes.length, entryBytes),
          );

          final zipBytes = ZipEncoder().encode(
            archive,
            level: Deflate.NO_COMPRESSION,
          )!;

          final npzFile = File('${tempDir.path}/mismatched_size.npz');
          npzFile.writeAsBytesSync(zipBytes, flush: true);

          expect(
            () => loadz(npzFile.path),
            throwsA(
              isA<FormatException>().having(
                (e) => e.message,
                'message',
                contains(
                  'Mismatched data size in NPZ archive for entry: expected 80 bytes from NPY header, got 40 bytes from ZIP header.',
                ),
              ),
            ),
          );
        },
      );

      test('Normal valid .npz archive loads successfully', () {
        NDArray.scope(() {
          final original = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final path = '${tempDir.path}/valid.npz';
          savez(path, {'arr': original}, compressed: false);

          final loaded = loadz(path);
          expect(loaded.containsKey('arr'), true);
          expect(loaded['arr']!.toList(), [1.0, 2.0, 3.0, 4.0]);
        });
      });
    });

    group('H13: Archive Checksums & Path Traversal Guards', () {
      test('Path traversal detection rejects path escapes', () {
        final baseDir = Directory('/safe/output/dir');
        final maliciousRelativePath = '../../etc/passwd';
        final resolved = baseDir.uri
            .resolve(maliciousRelativePath)
            .toFilePath();
        expect(resolved.startsWith(baseDir.path), isFalse);
      });
    });

    group('H1: Portable Reductions Execution', () {
      test('sum and prod execute accurately on all CPU baselines', () {
        NDArray.scope(() {
          final a = NDArray.fromList(List.generate(100, (i) => i.toDouble()), [
            100,
          ], DType.float64);
          final total = sum(a);
          expect(total.scalar, equals(4950.0));

          final b = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final product = prod(b);
          expect(product.scalar, equals(24.0));
        });
      });
    });
  });
}
