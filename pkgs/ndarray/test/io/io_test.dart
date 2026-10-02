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
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_extensions_bindings.dart';
import 'package:test/test.dart';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

/// Root for files written on dart2wasm, where `dart:io` is unavailable.
///
/// `tool/build_wasm.dart` passes a per-target directory and deletes it after
/// the run; the default only applies when a test is run some other way.
const String _wasmTempRoot = String.fromEnvironment(
  'NDARRAY_TEST_TMPDIR',
  defaultValue: '/tmp',
);

void main() {
  Directory? tempDir;
  late String tempDirPath;
  group('NDArray NumPy Binary Interoperability & I/O Tests', () {
    setUpAll(() {
      if (_isWasm) {
        // save()/savez() create missing parent directories natively.
        tempDirPath =
            '$_wasmTempRoot/ndarray_io_test_${DateTime.now().microsecondsSinceEpoch}';
      } else {
        tempDir = Directory.systemTemp.createTempSync('ndarray_io_test_');
        tempDirPath = tempDir!.path;
      }
    });

    tearDownAll(() {
      if (!_isWasm && tempDir != null && tempDir!.existsSync()) {
        tempDir!.deleteSync(recursive: true);
      }
    });

    group('.npy Round-Trip Tests', () {
      test(
        'Float64 2D Matrix round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
            [2, 3],
            DType.float64,
          );

          final path = '$tempDirPath/test_f64.npy';
          save(path, a);

          final loaded = load(path);
          expect(loaded.shape, [2, 3]);
          expect(loaded.dtype, DType.float64);
          expect(loaded.toList(), [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
        }),
      );

      test(
        'Float32 1D Vector round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([-1.0, 0.0, 1.0, 10.5]),
            [4],
            DType.float32,
          );
          final path = '$tempDirPath/test_f32.npy';
          save(path, a);

          final loaded = load(path);
          expect(loaded.shape, [4]);
          expect(loaded.dtype, DType.float32);
          expect(loaded.toList(), [-1.0, 0.0, 1.0, 10.5]);
        }),
      );

      test(
        'Int32 matrix round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([10, 20, 30, 40]), [
            2,
            2,
          ], DType.int32);
          final path = '$tempDirPath/test_i32.npy';
          save(path, a);

          final loaded = load(path);
          expect(loaded.dtype, DType.int32);
          expect(loaded.toList(), [10, 20, 30, 40]);
        }),
      );

      test(
        'Int64 vector round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Int64List.fromList([999999999, 111111111]),
            [2],
            DType.int64,
          );
          final path = '$tempDirPath/test_i64.npy';
          save(path, a);

          final loaded = load(path);
          expect(loaded.dtype, DType.int64);
          expect(loaded.toList(), [999999999, 111111111]);
        }),
      );

      test(
        'Complex128 array round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1.0, -2.0), Complex(0.0, 3.5)],
            [2],
            DType.complex128,
          );

          final path = '$tempDirPath/test_c16.npy';
          save(path, a);

          final loaded = load(path);
          expect(loaded.shape, [2]);
          expect(loaded.dtype, DType.complex128);
          final loadedList = loaded.toList();
          expect(loadedList[0], Complex(1.0, -2.0));
          expect(loadedList[1], Complex(0.0, 3.5));
        }),
      );

      test(
        'Complex64 array round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1.5, -2.5), Complex(0.0, 3.0)],
            [2],
            DType.complex64,
          );

          final path = '$tempDirPath/test_c8.npy';
          save(path, a);

          final loaded = load(path);
          expect(loaded.shape, [2]);
          expect(loaded.dtype, DType.complex64);
          final loadedList = loaded.toList();
          expect(loadedList[0], Complex(1.5, -2.5));
          expect(loadedList[1], Complex(0.0, 3.0));
        }),
      );

      test(
        'Non-contiguous view save creates contiguous file copy seamlessly',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );

          // Transposed view is non-contiguous (strides: [1, 2])
          final view = parent.transpose();
          expect(view.isContiguous, false);

          final path = '$tempDirPath/test_view.npy';
          save(path, view); // should make contiguous copy in-flight

          final loaded = load(path);
          expect(loaded.shape, [2, 2]);
          // Transposed data logic: 1, 3, 2, 4
          expect(loaded.toList(), [1.0, 3.0, 2.0, 4.0]);
        }),
      );
    });

    group('.npz Multi-Array Archive Tests', () {
      test(
        'Save and Load uncompressed .npz archive map',
        () => NDArray.scope(() {
          final arr1 = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
            2,
          ], DType.float64);
          final arr2 = NDArray.fromList(Int32List.fromList([5, 6, 7, 8]), [
            2,
            2,
          ], DType.int32);

          final map = {'array_one': arr1, 'array_two': arr2};

          final path = '$tempDirPath/archive.npz';
          savez(path, map, compressed: false);

          final loaded = loadz(path);
          expect(loaded.containsKey('array_one'), true);
          expect(loaded.containsKey('array_two'), true);

          expect(loaded['array_one']!.toList(), [1.0, 2.0]);
          expect(loaded['array_two']!.toList(), [5, 6, 7, 8]);
          expect(loaded['array_two']!.shape, [2, 2]);
        }),
      );

      test(
        'Save and Load compressed .npz archive map',
        () => NDArray.scope(() {
          final arr1 = NDArray.fromList(Float32List.fromList([0.5, 1.5]), [
            2,
          ], DType.float32);
          final path = '$tempDirPath/archive_comp.npz';
          savez(path, {'x': arr1}, compressed: true);

          final loaded = loadz(path);
          expect(loaded['x']!.toList(), [0.5, 1.5]);
          expect(loaded['x']!.dtype, DType.float32);
        }),
      );

      test(
        'Load Fortran ordered .npz archive map simulated from Python',
        () => NDArray.scope(() {
          // Build a fake in-memory .npy byte buffer that flags 'fortran_order': True
          final descr = _dtypeToDescr(DType.float64);
          final headerStr =
              "{'descr': '$descr', 'fortran_order': True, 'shape': (2, 3)}";

          final prefixLen = 6 + 2 + 2;
          final paddedHeaderLen =
              ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
          final padCount = paddedHeaderLen - headerStr.length - 1;
          final paddedHeader = "$headerStr${' ' * padCount}\n";

          final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
          final lenBytes = Uint8List(2);
          ByteData.view(
            lenBytes.buffer,
          ).setUint16(0, headerBytes.length, Endian.little);

          final rawData = Float64List.fromList([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
          final rawDataBytes = Uint8List.view(rawData.buffer);

          final fullBuffer = Uint8List(
            6 + 2 + 2 + headerBytes.length + rawDataBytes.length,
          );
          var offset = 0;

          fullBuffer.setRange(offset, offset + 6, const [
            0x93,
            0x4e,
            0x55,
            0x4d,
            0x50,
            0x59,
          ]);
          offset += 6;
          fullBuffer.setRange(offset, offset + 2, const [0x01, 0x00]);
          offset += 2;
          fullBuffer.setRange(offset, offset + 2, lenBytes);
          offset += 2;
          fullBuffer.setRange(offset, offset + headerBytes.length, headerBytes);
          offset += headerBytes.length;
          fullBuffer.setRange(
            offset,
            offset + rawDataBytes.length,
            rawDataBytes,
          );

          // Pack this Fortran npy buffer inside a zip archive
          final archive = Archive();
          archive.addFile(
            ArchiveFile('f_arr.npy', fullBuffer.length, fullBuffer),
          );
          final encoder = ZipEncoder();
          final zipBytes = encoder.encode(
            archive,
            level: Deflate.NO_COMPRESSION,
          )!;

          final path = '$tempDirPath/archive_fortran_simulated.npz';
          File(path).writeAsBytesSync(zipBytes, flush: true);

          // Load the archive
          final loaded = loadz(path);
          expect(loaded.containsKey('f_arr'), true);
          expect(loaded['f_arr']!.shape, [2, 3]);
          // Check that loaded array successfully restores strides to Column-Major!
          expect(loaded['f_arr']!.strides, [1, 2]);
          expect(loaded['f_arr']!.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );
    });

    group('Fortran Column-Major Layout Simulation Tests', () {
      test(
        'Simulate and parse a Python-generated fortran_order=True header file',
        () => NDArray.scope(() {
          // We will build a fake .npy in-memory byte buffer that flags 'fortran_order': True.
          // For a 2x3 matrix [[1, 2, 3], [4, 5, 6]], the Column-Major flat ordering in memory
          // is: [1, 4, 2, 5, 3, 6]!
          final descr = _dtypeToDescr(DType.float64);
          final headerStr =
              "{'descr': '$descr', 'fortran_order': True, 'shape': (2, 3)}";

          final prefixLen = 6 + 2 + 2;
          final paddedHeaderLen =
              ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
          final padCount = paddedHeaderLen - headerStr.length - 1;
          final paddedHeader = "$headerStr${' ' * padCount}\n";

          final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
          final lenBytes = Uint8List(2);
          ByteData.view(
            lenBytes.buffer,
          ).setUint16(0, headerBytes.length, Endian.little);

          // 6 doubles for data [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
          final rawData = Float64List.fromList([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
          final rawDataBytes = Uint8List.view(rawData.buffer);

          final fullBuffer = Uint8List(
            6 + 2 + 2 + headerBytes.length + rawDataBytes.length,
          );
          var offset = 0;

          fullBuffer.setRange(offset, offset + 6, const [
            0x93,
            0x4e,
            0x55,
            0x4d,
            0x50,
            0x59,
          ]);
          offset += 6;
          fullBuffer.setRange(offset, offset + 2, const [0x01, 0x00]);
          offset += 2;
          fullBuffer.setRange(offset, offset + 2, lenBytes);
          offset += 2;
          fullBuffer.setRange(offset, offset + headerBytes.length, headerBytes);
          offset += headerBytes.length;
          fullBuffer.setRange(
            offset,
            offset + rawDataBytes.length,
            rawDataBytes,
          );

          // Write this fake file to disk
          final path = '$tempDirPath/fortran_simulated.npy';
          File(path).writeAsBytesSync(fullBuffer, flush: true);

          // Load it via ndarray load()!
          final loaded = load(path);

          expect(loaded.shape, [2, 3]);
          // The zero-copy Fortran strides must be exactly: [1, 2]!
          expect(loaded.strides, [1, 2]);

          // Under stride-view indexing, loaded[i, j] translates to data[i*strides[0] + j*strides[1]].
          // loaded[0, 0] -> data[0] = 1.0
          // loaded[0, 1] -> data[2] = 2.0
          // loaded[0, 2] -> data[4] = 3.0
          // loaded[1, 0] -> data[1] = 4.0
          // loaded[1, 1] -> data[3] = 5.0
          // loaded[1, 2] -> data[5] = 6.0
          // So calling toList() (which loops row-major logically) should yield exactly [1, 2, 3, 4, 5, 6]!
          expect(loaded.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
          // Success! Stride reindexing mapping loaded column-major binary files with absolute zero data copies!
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );
    });

    group('Error / Exception Handlers Tests', () {
      test(
        'Non-existent files throw FileSystemException in load and loadz',
        () {
          expect(
            () => load('$tempDirPath/non_existent_file.npy'),
            throwsA(isA<FileSystemException>()),
          );
          expect(
            () => loadz('$tempDirPath/non_existent_archive.npz'),
            throwsA(isA<FileSystemException>()),
          );
        },
      );

      test(
        'Invalid Magic signature in load() throws FormatException',
        () => NDArray.scope(() {
          final file = File('$tempDirPath/corrupted.npy');
          file.writeAsBytesSync([
            0x00,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
          ]);
          expect(
            () => load('$tempDirPath/corrupted.npy'),
            throwsFormatException,
          );
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'Big-endian header descriptor throws UnsupportedError',
        () => NDArray.scope(() {
          final path = '$tempDirPath/big_endian_simulated.npy';
          _writeFakeNpy(
            path,
            "{'descr': '>f8', 'fortran_order': False, 'shape': (2,)}",
          );
          expect(() => load(path), throwsUnsupportedError);
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'Unsupported NumPy descriptor throws UnsupportedError',
        () => NDArray.scope(() {
          final path = '$tempDirPath/bad_descr.npy';
          _writeFakeNpy(
            path,
            "{'descr': '<f16', 'fortran_order': False, 'shape': (2,)}",
          );
          expect(() => load(path), throwsUnsupportedError);
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'Missing descr in header throws FormatException',
        () => NDArray.scope(() {
          final path = '$tempDirPath/missing_descr.npy';
          _writeFakeNpy(path, "{'fortran_order': False, 'shape': (2,)}");
          expect(() => load(path), throwsFormatException);
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'Missing fortran_order in header throws FormatException',
        () => NDArray.scope(() {
          final path = '$tempDirPath/missing_fortran.npy';
          _writeFakeNpy(path, "{'descr': '<f8', 'shape': (2,)}");
          expect(() => load(path), throwsFormatException);
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'Missing shape in header throws FormatException',
        () => NDArray.scope(() {
          final path = '$tempDirPath/missing_shape.npy';
          _writeFakeNpy(path, "{'descr': '<f8', 'fortran_order': False}");
          expect(() => load(path), throwsFormatException);
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'Short npy file lacking format version headers throws FormatException',
        () {
          final file = File('$tempDirPath/short_version.npy');
          file.writeAsBytesSync([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01]);
          expect(
            () => load('$tempDirPath/short_version.npy'),
            throwsFormatException,
          );
        },
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );
      test(
        'load() throws FormatException when header lacks "descr" parameter',
        () {
          _writeFakeNpy(
            '$tempDirPath/missing_descr.npy',
            "{'fortran_order': False, 'shape': (2, 2)}",
          );
          expect(
            () => load('$tempDirPath/missing_descr.npy'),
            throwsFormatException,
          );
        },
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'load() throws UnsupportedError when descriptor is unsupported',
        () => NDArray.scope(() {
          _writeFakeNpy(
            '$tempDirPath/unsupported_dtype.npy',
            "{'descr': '<f16', 'fortran_order': False, 'shape': (2, 2)}",
          );
          expect(
            () => load('$tempDirPath/unsupported_dtype.npy'),
            throwsUnsupportedError,
          );
        }),
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        '_deserializeNpyBytes throws FormatException on invalid magic bytes',
        () {
          final badBytes = Uint8List.fromList([
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
          ]);
          final archive = Archive();
          archive.addFile(
            ArchiveFile('corrupted.npy', badBytes.length, badBytes),
          );
          final encoder = ZipEncoder();
          final zipBytes = encoder.encode(
            archive,
            level: Deflate.NO_COMPRESSION,
          )!;

          final path = '$tempDirPath/bad_archive_magic.npz';
          File(path).writeAsBytesSync(zipBytes, flush: true);

          expect(() => loadz(path), throwsFormatException);
        },
        skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
      );

      test(
        'loadz() throws FormatException when NPZ entry fails CRC32 check',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final path = '$tempDirPath/corrupted_crc.npz';
          savez(path, {'a': a}, compressed: false);

          // Flip a byte inside the raw float64 payload (leaving ZIP headers and CRC32 intact)
          final bytes = File(path).readAsBytesSync();
          // Find the NPY magic inside the ZIP and corrupt the last byte of the 32-byte float64 payload
          for (var i = 0; i < bytes.length - 6; i++) {
            if (bytes[i] == 0x93 &&
                bytes[i + 1] == 0x4e &&
                bytes[i + 2] == 0x55 &&
                bytes[i + 3] == 0x4d &&
                bytes[i + 4] == 0x50 &&
                bytes[i + 5] == 0x59) {
              final hlen = bytes[i + 8] | (bytes[i + 9] << 8);
              final dataStart = i + 10 + hlen;
              bytes[dataStart] ^= 0xff;
              break;
            }
          }
          File(path).writeAsBytesSync(bytes, flush: true);

          expect(() => loadz(path), throwsFormatException);
        }),
        skip: _isWasm ? 'Uses dart:io File.readAsBytesSync' : false,
      );
    });

    group('Additional I/O Coverage Tests', () {
      test(
        'Save into brand new nested directory makes parent directory recursive',
        () => NDArray.scope(() {
          final a = NDArray.ones([2], DType.float64);
          final path = '$tempDirPath/nested_non_existent/nested_level/arr.npy';
          save(path, a);

          final file = File(path);
          expect(file.existsSync(), true);
          final loaded = load(path);
          expect(loaded.toList(), [1.0, 1.0]);

          file.deleteSync();
          Directory(
            '$tempDirPath/nested_non_existent',
          ).deleteSync(recursive: true);
        }),
        skip: _isWasm ? 'Uses dart:io Directory.createSync' : false,
      );

      test(
        'Save non-contiguous view inside savez archive map',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final view = parent.transposed;
          expect(view.isContiguous, false);

          final path = '$tempDirPath/archive_with_view.npz';
          savez(path, {'view_key': view}, compressed: false);

          final loaded = loadz(path);
          expect(loaded.containsKey('view_key'), true);
          expect(loaded['view_key']!.toList(), [1.0, 3.0, 2.0, 4.0]);

          if (!_isWasm) {
            File(path).deleteSync();
          }
        }),
      );

      test(
        'Savez npz file into brand new nested directory makes parent directory recursive',
        () => NDArray.scope(() {
          final a = NDArray.ones([2], DType.float64);
          final path = '$tempDirPath/nested_npz_dir/nested_level/archive.npz';
          savez(path, {'arr': a}, compressed: false);

          final file = File(path);
          expect(file.existsSync(), true);
          final loaded = loadz(path);
          expect(loaded['arr']!.toList(), [1.0, 1.0]);

          file.deleteSync();
          Directory('$tempDirPath/nested_npz_dir').deleteSync(recursive: true);
        }),
        skip: _isWasm ? 'Uses dart:io Directory.createSync' : false,
      );
    });
    test(
      'NPY double quotes / mixed quotes header parsing compatibility',
      () => NDArray.scope(() {
        // Construct a fake .npy buffer using double quotes in header dictionary
        final headerStr =
            '{"descr": "<f8", "fortran_order": False, "shape": (2, 2)}';

        final prefixLen = 6 + 2 + 2;
        final paddedHeaderLen =
            ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
        final padCount = paddedHeaderLen - headerStr.length - 1;
        final paddedHeader = "$headerStr${' ' * padCount}\n";

        final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
        final lenBytes = Uint8List(2);
        ByteData.view(
          lenBytes.buffer,
        ).setUint16(0, headerBytes.length, Endian.little);

        final rawData = Float64List.fromList([1.0, 2.0, 3.0, 4.0]);
        final rawDataBytes = Uint8List.view(rawData.buffer);

        final fullBuffer = Uint8List(
          6 + 2 + 2 + headerBytes.length + rawDataBytes.length,
        );
        var offset = 0;

        fullBuffer.setRange(offset, offset + 6, const [
          0x93,
          0x4e,
          0x55,
          0x4d,
          0x50,
          0x59,
        ]);
        offset += 6;
        fullBuffer.setRange(offset, offset + 2, const [0x01, 0x00]);
        offset += 2;
        fullBuffer.setRange(offset, offset + 2, lenBytes);
        offset += 2;
        fullBuffer.setRange(offset, offset + headerBytes.length, headerBytes);
        offset += headerBytes.length;
        fullBuffer.setRange(offset, offset + rawDataBytes.length, rawDataBytes);

        final path = '$tempDirPath/double_quotes_simulated.npy';
        File(path).writeAsBytesSync(fullBuffer, flush: true);

        // Load should parse successfully now
        final loaded = load(path);
        expect(loaded.shape, [2, 2]);
        expect(loaded.dtype, DType.float64);
        expect(loaded.toList(), [1.0, 2.0, 3.0, 4.0]);
      }),
      skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false,
    );

    test(
      'save() and load() non-contiguous strided views of all dtypes',
      () => NDArray.scope(() {
        // 1. float32
        final f32 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final f32View = f32.transposed; // non-contiguous!
        save('$tempDirPath/f32_view.npy', f32View);
        final f32Loaded = load('$tempDirPath/f32_view.npy');
        expect(f32Loaded.shape, [2, 2]);
        expect(f32Loaded.dtype, DType.float32);
        expect(f32Loaded.toList(), [1.0, 3.0, 2.0, 4.0]);

        // 2. int32
        final i32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32View = i32.transposed;
        save('$tempDirPath/i32_view.npy', i32View);
        final i32Loaded = load('$tempDirPath/i32_view.npy');
        expect(i32Loaded.toList(), [1, 3, 2, 4]);

        // 3. int64
        final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
        final i64View = i64.transposed;
        save('$tempDirPath/i64_view.npy', i64View);
        final i64Loaded = load('$tempDirPath/i64_view.npy');
        expect(i64Loaded.toList(), [1, 3, 2, 4]);

        // 4. boolean
        final b = NDArray.fromList(
          [true, false, true, false],
          [2, 2],
          DType.boolean,
        );
        final bView = b.transposed;
        save('$tempDirPath/b_view.npy', bView);
        final bLoaded = load('$tempDirPath/b_view.npy');
        expect(bLoaded.toList(), [true, true, false, false]);

        // 5. complex128
        final c128 = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );
        final c128View = c128.transposed;
        save('$tempDirPath/c128_view.npy', c128View);
        final c128Loaded = load('$tempDirPath/c128_view.npy');
        expect(c128Loaded.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);

        // 6. complex64
        final c64 = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex64,
        );
        final c64View = c64.transposed;
        save('$tempDirPath/c64_view.npy', c64View);
        final c64Loaded = load('$tempDirPath/c64_view.npy');
        expect(c64Loaded.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);
      }),
    );

    test(
      'save() and load() Uint8 and Int16 arrays coverage',
      () => NDArray.scope(() {
        // 1. Uint8
        final u8 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
        save('$tempDirPath/u8_array.npy', u8);
        final u8Loaded = load('$tempDirPath/u8_array.npy');
        expect(u8Loaded.toList(), [1, 2, 3, 4]);
        expect(u8Loaded.dtype, DType.uint8);

        // 2. Int16
        final i16 = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int16);
        save('$tempDirPath/i16_array.npy', i16);
        final i16Loaded = load('$tempDirPath/i16_array.npy');
        expect(i16Loaded.toList(), [10, 20, 30, 40]);
        expect(i16Loaded.dtype, DType.int16);
      }),
    );
    group('Adversarial & Malformed NPY / NPZ Header Robustness', () {
      test(
        'Negative shape dimension in .npy throws FormatException',
        () => NDArray.scope(() {
          final path1 = '$tempDirPath/negative_dim_2d.npy';
          _writeFakeNpy(
            path1,
            "{'descr': '<f8', 'fortran_order': False, 'shape': (-1, 4)}",
          );
          expect(() => load(path1), throwsFormatException);

          final path2 = '$tempDirPath/negative_dim_1d.npy';
          _writeFakeNpy(
            path2,
            "{'descr': '<f8', 'fortran_order': False, 'shape': (-5,)}",
          );
          expect(() => load(path2), throwsFormatException);
        }),
      );

      test(
        'Overflowing shape dimension (> 2^31 - 1 elements) in .npy throws cleanly',
        () => NDArray.scope(() {
          final path = '$tempDirPath/overflow_dim.npy';
          _writeFakeNpy(
            path,
            "{'descr': '<f8', 'fortran_order': False, 'shape': (3000000000,)}",
          );
          expect(
            () => load(path),
            throwsA(
              anyOf(
                isA<FormatException>(),
                isA<ArgumentError>(),
                isA<UnsupportedError>(),
              ),
            ),
          );
        }),
      );

      test(
        'Overflowing shape 64-bit product in .npy throws cleanly',
        () => NDArray.scope(() {
          final path1 = '$tempDirPath/product_overflow_64.npy';
          _writeFakeNpy(
            path1,
            "{'descr': '<f8', 'fortran_order': False, 'shape': (3037000500, 3037000500)}",
          );
          expect(
            () => load(path1),
            throwsA(
              anyOf(
                isA<FormatException>(),
                isA<ArgumentError>(),
                isA<UnsupportedError>(),
              ),
            ),
          );

          final path2 = '$tempDirPath/product_overflow_31.npy';
          _writeFakeNpy(
            path2,
            "{'descr': '<f8', 'fortran_order': False, 'shape': (50000, 50000)}",
          );
          expect(
            () => load(path2),
            throwsA(
              anyOf(
                isA<FormatException>(),
                isA<ArgumentError>(),
                isA<UnsupportedError>(),
              ),
            ),
          );
        }),
      );

      test('Truncated .npy payload throws FormatException', () {
        final path = '$tempDirPath/truncated_payload.npy';
        _writeFakeNpy(
          path,
          "{'descr': '<f8', 'fortran_order': False, 'shape': (10,)}",
          payloadBytes: Uint8List(16),
        );
        expect(() => load(path), throwsFormatException);
      });

      test(
        'Negative shape dimension in .npz archive entry throws FormatException',
        () {
          final npzPath = '$tempDirPath/corrupt_negative_dim.npz';
          final npyBytes = _buildFakeNpyBytes(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (-1, 4)}",
            payloadBytes: Uint8List(32),
          );
          final archive = Archive();
          archive.addFile(ArchiveFile('bad.npy', npyBytes.length, npyBytes));
          File(npzPath).writeAsBytesSync(ZipEncoder().encode(archive)!);

          expect(() => loadz(npzPath), throwsFormatException);
        },
      );

      test('Overflowing shape in .npz archive entry throws cleanly', () {
        final npzPath = '$tempDirPath/corrupt_overflow_dim.npz';
        final npyBytes = _buildFakeNpyBytes(
          "{'descr': '<f8', 'fortran_order': False, 'shape': (3037000500, 3037000500)}",
          payloadBytes: Uint8List(16),
        );
        final archive = Archive();
        archive.addFile(ArchiveFile('bad.npy', npyBytes.length, npyBytes));
        File(npzPath).writeAsBytesSync(ZipEncoder().encode(archive)!);

        expect(
          () => loadz(npzPath),
          throwsA(
            anyOf(
              isA<FormatException>(),
              isA<ArgumentError>(),
              isA<UnsupportedError>(),
            ),
          ),
        );
      });

      test(
        'Truncated payload in .npz archive entry throws FormatException',
        () {
          final npzPath = '$tempDirPath/corrupt_truncated_payload.npz';
          final npyBytes = _buildFakeNpyBytes(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (5,)}",
            payloadBytes: Uint8List(8),
          );
          final archive = Archive();
          archive.addFile(ArchiveFile('bad.npy', npyBytes.length, npyBytes));
          File(npzPath).writeAsBytesSync(ZipEncoder().encode(archive)!);

          expect(() => loadz(npzPath), throwsFormatException);
        },
      );
    }, skip: _isWasm ? 'Uses dart:io File.writeAsBytesSync' : false);

    group('ZIP64 .npz Archive Support (> 4 GiB Format)', () {
      test(
        'ZIP64 STORED (uncompressed) .npz writes 0x0001 extra fields, ZIP64 EOCD (0x06064b50), and round-trips via loadz',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [2, 3],
            DType.float64,
          );
          final b = NDArray.fromList([-1, 2, -3, 4], [4], DType.int64);
          final path = '$tempDirPath/zip64_stored.npz';
          _saveNpzWithFlags(path, {'alpha': a, 'beta': b}, 0x100);

          if (!_isWasm) {
            final rawBytes = File(path).readAsBytesSync();
            // Verify ZIP64 End of Central Directory Record (0x06064b50) and
            // ZIP64 End of Central Directory Locator (0x07064b50) signatures exist.
            expect(_containsU32Le(rawBytes, 0x06064b50), isTrue);
            expect(_containsU32Le(rawBytes, 0x07064b50), isTrue);
          }

          final loaded = loadz(path);
          expect(loaded.keys.toSet(), {'alpha', 'beta'});
          expect(loaded['alpha']!.shape, [2, 3]);
          expect(loaded['alpha']!.dtype, DType.float64);
          expect(loaded['alpha']!.toList(), [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
          ]);
          expect(loaded['beta']!.shape, [4]);
          expect(loaded['beta']!.dtype, DType.int64);
          expect(loaded['beta']!.toList(), [-1, 2, -3, 4]);
        }),
      );

      test(
        'ZIP64 DEFLATE (compressed) .npz writes ZIP64 EOCD and round-trips via loadz',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            List<double>.generate(128, (i) => i * 1.25),
            [16, 8],
            DType.float64,
          );
          final path = '$tempDirPath/zip64_deflate.npz';
          _saveNpzWithFlags(path, {'matrix': a}, 0x106);

          if (!_isWasm) {
            final rawBytes = File(path).readAsBytesSync();
            expect(_containsU32Le(rawBytes, 0x06064b50), isTrue);
            expect(_containsU32Le(rawBytes, 0x07064b50), isTrue);
          }

          final loaded = loadz(path);
          expect(loaded.keys.toSet(), {'matrix'});
          expect(loaded['matrix']!.shape, [16, 8]);
          expect(loaded['matrix']!.dtype, DType.float64);
          expect(
            loaded['matrix']!.toList(),
            List<double>.generate(128, (i) => i * 1.25),
          );
        }),
      );
    });

    group('Edge-case arrays and error contracts', () {
      test('.npy roundtrip preserves float16 and bfloat16 dtypes', () {
        NDArray.scope(() {
          final f16 = NDArray.fromList([0.5, -1.5, 2.25], [3], DType.float16);
          final pathF16 = '$tempDirPath/half_f16.npy';
          save(pathF16, f16);
          final loadedF16 = load(pathF16);
          expect(loadedF16.dtype, equals(DType.float16));
          expect(loadedF16.toList(), equals(f16.toList()));

          final bf16 = NDArray.fromList([1.0, -2.0, 4.0], [3], DType.bfloat16);
          final pathBf16 = '$tempDirPath/half_bf16.npy';
          save(pathBf16, bf16);
          final loadedBf16 = load(pathBf16);
          expect(loadedBf16.dtype, equals(DType.bfloat16));
          expect(loadedBf16.toList(), equals(bf16.toList()));
        });
      });

      test('0-D scalar and empty [0, 3] array .npy roundtrip', () {
        NDArray.scope(() {
          final scalar = NDArray.scalar(-42.5, dtype: DType.float64);
          final pathScalar = '$tempDirPath/edge_scalar.npy';
          save(pathScalar, scalar);
          final loadedScalar = load(pathScalar) as NDArray<Float64>;
          expect(loadedScalar.rank, equals(0));
          expect(loadedScalar.scalar, equals(-42.5));

          final empty = NDArray.zeros([0, 3], DType.int64);
          final pathEmpty = '$tempDirPath/edge_empty.npy';
          save(pathEmpty, empty);
          final loadedEmpty = load(pathEmpty) as NDArray<Int64>;
          expect(loadedEmpty.shape, equals([0, 3]));
          expect(loadedEmpty.size, equals(0));
        });
      });

      test(
        'negative-stride flipped view .npy and compressed .npz roundtrip',
        () {
          NDArray.scope(() {
            final base = NDArray.arange(
              0,
              6,
              dtype: DType.float64,
            ).reshape([2, 3]);
            final flipped = flip(base);
            expect(flipped.isContiguous, isFalse);

            final npyPath = '$tempDirPath/edge_flipped.npy';
            save(npyPath, flipped);
            final loadedNpy = load(npyPath) as NDArray<Float64>;
            expect(loadedNpy.shape, equals([2, 3]));
            expect(loadedNpy.toList(), equals([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]));

            final npzPath = '$tempDirPath/edge_flipped.npz';
            savez(npzPath, {'flipped': flipped}, compressed: true);
            final loadedNpz = loadz(npzPath);
            expect(
              loadedNpz['flipped']!.toList(),
              equals([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
            );
          });
        },
      );

      test('saving a disposed array throws StateError', () {
        final a = NDArray.ones([2], DType.float64);
        a.dispose();
        expect(
          () => save('$tempDirPath/edge_disposed.npy', a),
          throwsStateError,
        );
        expect(
          () => savez('$tempDirPath/edge_disposed.npz', {'a': a}),
          throwsStateError,
        );
      });

      test('loadz on a plain .npy file throws FormatException', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final npyPath = '$tempDirPath/edge_not_a_zip.npy';
          save(npyPath, a);
          expect(() => loadz(npyPath), throwsFormatException);
        });
      });
    });
  });
}

bool _containsU32Le(Uint8List bytes, int target) {
  final b0 = target & 0xFF;
  final b1 = (target >> 8) & 0xFF;
  final b2 = (target >> 16) & 0xFF;
  final b3 = (target >> 24) & 0xFF;
  for (var i = 0; i + 4 <= bytes.length; i++) {
    if (bytes[i] == b0 &&
        bytes[i + 1] == b1 &&
        bytes[i + 2] == b2 &&
        bytes[i + 3] == b3) {
      return true;
    }
  }
  return false;
}

void _setPtrAt<T extends ffi.NativeType>(
  ffi.Pointer<ffi.Pointer<T>> array,
  int index,
  ffi.Pointer<T> value,
) {
  if (ffi.sizeOf<ffi.IntPtr>() == 4) {
    array.cast<ffi.Uint32>()[index] = value.address;
  } else {
    array.cast<ffi.Uint64>()[index] = value.address;
  }
}

void _setSizeAt(ffi.Pointer<ffi.Size> array, int index, int value) {
  if (ffi.sizeOf<ffi.Size>() == 4) {
    array.cast<ffi.Uint32>()[index] = value;
  } else {
    array.cast<ffi.Uint64>()[index] = value;
  }
}

void _saveNpzWithFlags(
  String filepath,
  Map<String, NDArray<DTypeTag>> arrays,
  int compressFlags,
) {
  final numArrays = arrays.length;
  final marker = ScratchArena.marker;
  try {
    final cNames = ScratchArena.allocate<ffi.Pointer<ffi.Char>>(
      numArrays * ffi.sizeOf<ffi.Pointer<ffi.Char>>(),
    );
    final cHeaderBytes = ScratchArena.allocate<ffi.Pointer<ffi.Uint8>>(
      numArrays * ffi.sizeOf<ffi.Pointer<ffi.Uint8>>(),
    );
    final cHeaderLens = ScratchArena.allocate<ffi.Size>(
      numArrays * ffi.sizeOf<ffi.Size>(),
    );
    final cDataPtrs = ScratchArena.allocate<ffi.Pointer<ffi.Void>>(
      numArrays * ffi.sizeOf<ffi.Pointer<ffi.Void>>(),
    );
    final cDataLens = ScratchArena.allocate<ffi.Size>(
      numArrays * ffi.sizeOf<ffi.Size>(),
    );

    ffi.Pointer<ffi.Char> allocUtf8(String s) {
      final units = utf8.encode(s);
      final ptr = ScratchArena.allocate<ffi.Uint8>(units.length + 1);
      for (var i = 0; i < units.length; i++) {
        ptr[i] = units[i];
      }
      ptr[units.length] = 0;
      return ptr.cast<ffi.Char>();
    }

    var idx = 0;
    for (final entry in arrays.entries) {
      final arr = entry.value;
      _setPtrAt(cNames, idx, allocUtf8('${entry.key}.npy'));
      final descr = arr.dtype.npyDescriptor;
      final shapeStr = arr.shape.length == 1
          ? '${arr.shape[0]},'
          : arr.shape.join(', ');
      final headerStr =
          "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";
      const prefixLen = 10;
      final paddedHeaderLen =
          ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
      final padCount = paddedHeaderLen - headerStr.length - 1;
      final paddedHeader = "$headerStr${' ' * padCount}\n";
      final headerCodeUnits = paddedHeader.codeUnits;
      final hLen = headerCodeUnits.length;
      final totalHeaderBytes = prefixLen + hLen;
      final hBuf = ScratchArena.allocate<ffi.Uint8>(totalHeaderBytes);
      hBuf[0] = 0x93;
      hBuf[1] = 0x4e;
      hBuf[2] = 0x55;
      hBuf[3] = 0x4d;
      hBuf[4] = 0x50;
      hBuf[5] = 0x59;
      hBuf[6] = 0x01;
      hBuf[7] = 0x00;
      hBuf[8] = hLen & 0xFF;
      hBuf[9] = (hLen >> 8) & 0xFF;
      for (var j = 0; j < hLen; j++) {
        hBuf[10 + j] = headerCodeUnits[j];
      }
      _setPtrAt(cHeaderBytes, idx, hBuf);
      _setSizeAt(cHeaderLens, idx, totalHeaderBytes);
      _setPtrAt(cDataPtrs, idx, arr.pointer.cast<ffi.Void>());
      _setSizeAt(cDataLens, idx, arr.size * arr.dtype.byteWidth);
      idx++;
    }

    final cFilepath = allocUtf8(filepath);
    final status = npz_save(
      cFilepath,
      numArrays,
      cNames,
      cHeaderBytes,
      cHeaderLens,
      cDataPtrs,
      cDataLens,
      compressFlags,
    );
    expect(status, 0);
  } finally {
    ScratchArena.reset(marker);
  }
}

Uint8List _buildFakeNpyBytes(
  String headerStr, {
  List<int> version = const [1, 0],
  List<int>? payloadBytes,
}) {
  final prefixLen = 6 + 2 + 2;
  final paddedHeaderLen =
      ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
  final padCount = paddedHeaderLen - headerStr.length - 1;
  final paddedHeader = "$headerStr${' ' * padCount}\n";

  final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
  final lenBytes = Uint8List(2);
  ByteData.view(
    lenBytes.buffer,
  ).setUint16(0, headerBytes.length, Endian.little);

  final payload = payloadBytes ?? Uint8List(16);
  final fullBuffer = Uint8List(6 + 2 + 2 + headerBytes.length + payload.length);
  fullBuffer.setRange(0, 6, const [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);
  fullBuffer.setRange(6, 8, version);
  fullBuffer.setRange(8, 10, lenBytes);
  fullBuffer.setRange(10, 10 + headerBytes.length, headerBytes);
  fullBuffer.setRange(10 + headerBytes.length, fullBuffer.length, payload);
  return fullBuffer;
}

void _writeFakeNpy(
  String path,
  String headerStr, {
  List<int> version = const [1, 0],
  List<int>? payloadBytes,
}) {
  final bytes = _buildFakeNpyBytes(
    headerStr,
    version: version,
    payloadBytes: payloadBytes,
  );
  File(path).writeAsBytesSync(bytes, flush: true);
}

// Simple helper function to map DType to descriptor string for test creation
String _dtypeToDescr(DType dtype) {
  switch (dtype) {
    case DType.float64:
      return '<f8';
    case DType.float32:
      return '<f4';
    case DType.int64:
      return '<i8';
    case DType.int32:
      return '<i4';
    case DType.complex128:
      return '<c16';
    case DType.complex64:
      return '<c8';
    case DType.boolean:
      return '|b1';
    default:
      throw UnimplementedError(
        'Unsupported dtype for test descriptor mapping: $dtype',
      );
  }
}
