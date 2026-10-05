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

import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_extensions_bindings.dart';
import 'package:test/test.dart';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

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

ffi.Pointer<ffi.Char> _allocCString(String s) {
  final units = utf8.encode(s);
  final ptr = ScratchArena.allocate<ffi.Uint8>(units.length + 1);
  for (var i = 0; i < units.length; i++) {
    ptr[i] = units[i];
  }
  ptr[units.length] = 0;
  return ptr.cast<ffi.Char>();
}

void _writeCustomNpz({
  required String filepath,
  required String entryName,
  required List<int> headerBytes,
  required ffi.Pointer<ffi.Void> dataPtr,
  required int dataByteLen,
  required bool compressed,
}) {
  final marker = ScratchArena.marker;
  try {
    final cNames = ScratchArena.allocate<ffi.Pointer<ffi.Char>>(
      ffi.sizeOf<ffi.Pointer<ffi.Char>>(),
    );
    final cHeaderBytes = ScratchArena.allocate<ffi.Pointer<ffi.Uint8>>(
      ffi.sizeOf<ffi.Pointer<ffi.Uint8>>(),
    );
    final cHeaderLens = ScratchArena.allocate<ffi.Size>(ffi.sizeOf<ffi.Size>());
    final cDataPtrs = ScratchArena.allocate<ffi.Pointer<ffi.Void>>(
      ffi.sizeOf<ffi.Pointer<ffi.Void>>(),
    );
    final cDataLens = ScratchArena.allocate<ffi.Size>(ffi.sizeOf<ffi.Size>());

    _setPtrAt(cNames, 0, _allocCString(entryName));
    final hBuf = ScratchArena.allocate<ffi.Uint8>(headerBytes.length);
    for (var i = 0; i < headerBytes.length; i++) {
      hBuf[i] = headerBytes[i];
    }
    _setPtrAt(cHeaderBytes, 0, hBuf);
    _setSizeAt(cHeaderLens, 0, headerBytes.length);
    _setPtrAt(cDataPtrs, 0, dataPtr);
    _setSizeAt(cDataLens, 0, dataByteLen);

    final cFilepath = _allocCString(filepath);
    final status = npz_save(
      cFilepath,
      1,
      cNames,
      cHeaderBytes,
      cHeaderLens,
      cDataPtrs,
      cDataLens,
      compressed ? 6 : 0,
    );
    if (status != 0) {
      throw StateError('npz_save failed with status $status');
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

List<int> _buildNpyV2Header({
  required String descr,
  required List<int> shape,
  required int targetHeaderPayloadLen,
}) {
  final shapeStr = shape.length == 1 ? '${shape[0]},' : shape.join(', ');
  final dictStr =
      "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";
  final padCount = targetHeaderPayloadLen - dictStr.length - 1;
  final paddedDict = '$dictStr${' ' * padCount}\n';
  final codeUnits = paddedDict.codeUnits;
  final hLen = codeUnits.length;

  return <int>[
    0x93,
    0x4e, // 'N'
    0x55, // 'U'
    0x4d, // 'M'
    0x50, // 'P'
    0x59, // 'Y'
    0x02, // major = 2
    0x00, // minor = 0
    hLen & 0xFF,
    (hLen >> 8) & 0xFF,
    (hLen >> 16) & 0xFF,
    (hLen >> 24) & 0xFF,
    ...codeUnits,
  ];
}

void main() {
  group('Issue #5: .npy v2.0 Header Support and Error Handling in loadz()', () {
    Directory? tempDir;
    late String tempDirPath;

    setUp(() {
      if (_isWasm) {
        tempDirPath = '/tmp';
      } else {
        tempDir = Directory.systemTemp.createTempSync('ndarray_cycle13_npz_');
        tempDirPath = tempDir!.path;
      }
    });

    tearDown(() {
      if (!_isWasm && tempDir != null && tempDir!.existsSync()) {
        tempDir!.deleteSync(recursive: true);
      }
    });

    for (final compressed in [false, true]) {
      test(
        'loadz() reads .npz (${compressed ? "compressed" : "uncompressed"}) with .npy v2.0 large header (> 65535 bytes)',
        () {
          NDArray.scope(() {
            final src = NDArray<Float64>.fromList(
              <double>[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
              [2, 3],
              DType.float64,
            );
            final headerBytes = _buildNpyV2Header(
              descr: '<f8',
              shape: [2, 3],
              targetHeaderPayloadLen: 70004,
            );
            expect(headerBytes.length, greaterThan(65536));

            final npzPath =
                '$tempDirPath/v2_large_${compressed ? "comp" : "uncomp"}.npz';
            _writeCustomNpz(
              filepath: npzPath,
              entryName: 'weights.npy',
              headerBytes: headerBytes,
              dataPtr: src.pointer.cast<ffi.Void>(),
              dataByteLen: 6 * 8,
              compressed: compressed,
            );

            final loaded = loadz(npzPath);
            try {
              expect(loaded.keys, contains('weights'));
              final arr = loaded['weights']!;
              expect(arr.dtype, equals(DType.float64));
              expect(arr.shape, equals([2, 3]));
              expect(arr.getCell([0, 0]), closeTo(1.5, 1e-12));
              expect(arr.getCell([0, 2]), closeTo(3.5, 1e-12));
              expect(arr.getCell([1, 2]), closeTo(6.5, 1e-12));
            } finally {
              for (final a in loaded.values) {
                a.dispose();
              }
            }
          });
        },
      );

      test(
        'loadz() reads .npz (${compressed ? "compressed" : "uncompressed"}) with .npy v2.0 compact header (<= 65535 bytes)',
        () {
          NDArray.scope(() {
            final src = NDArray<Int32>.fromList(
              <int>[10, 20, 30, 40],
              [4],
              DType.int32,
            );
            final headerBytes = _buildNpyV2Header(
              descr: '<i4',
              shape: [4],
              targetHeaderPayloadLen: 116,
            );

            final npzPath =
                '$tempDirPath/v2_compact_${compressed ? "comp" : "uncomp"}.npz';
            _writeCustomNpz(
              filepath: npzPath,
              entryName: 'vec.npy',
              headerBytes: headerBytes,
              dataPtr: src.pointer.cast<ffi.Void>(),
              dataByteLen: 4 * 4,
              compressed: compressed,
            );

            final loaded = loadz(npzPath);
            try {
              expect(loaded.keys, contains('vec'));
              final arr = loaded['vec']!;
              expect(arr.dtype, equals(DType.int32));
              expect(arr.shape, equals([4]));
              expect(arr.getCell([0]), equals(10));
              expect(arr.getCell([3]), equals(40));
            } finally {
              for (final a in loaded.values) {
                a.dispose();
              }
            }
          });
        },
      );

      test(
        'loadz() throws FormatException on truncated/corrupted .npy entry (${compressed ? "compressed" : "uncompressed"}) instead of silently dropping it',
        () {
          NDArray.scope(() {
            final dummy = NDArray<Int32>.fromList(<int>[1], [1], DType.int32);

            // 1. Entry truncated to < 10 bytes (status -7)
            final shortPath =
                '$tempDirPath/corrupt_short_${compressed ? "comp" : "uncomp"}.npz';
            _writeCustomNpz(
              filepath: shortPath,
              entryName: 'broken.npy',
              headerBytes: <int>[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59],
              dataPtr: dummy.pointer.cast<ffi.Void>(),
              dataByteLen: 0,
              compressed: compressed,
            );
            expect(() => loadz(shortPath), throwsFormatException);

            // 2. Entry whose header length exceeds uncompressed size (status -11)
            final overflowPath =
                '$tempDirPath/corrupt_hlen_${compressed ? "comp" : "uncomp"}.npz';
            final badHeader = <int>[
              0x93,
              0x4e,
              0x55,
              0x4d,
              0x50,
              0x59,
              0x02,
              0x00,
              0x00,
              0x10,
              0x00,
              0x00, // claims hlen = 4096 bytes, but only 4 bytes follow
            ];
            _writeCustomNpz(
              filepath: overflowPath,
              entryName: 'bad_hlen.npy',
              headerBytes: badHeader,
              dataPtr: dummy.pointer.cast<ffi.Void>(),
              dataByteLen: 4,
              compressed: compressed,
            );
            expect(() => loadz(overflowPath), throwsFormatException);
          });
        },
      );
    }
  });

  group('Issue #6: _sliceAssignImpl DType Coercion & Fancy Indexing Broadcast', () {
    test(
      'arr[Slice(start: 0, stop: 2)] = int32Arr and arr.sliceAssign([Slice(start: 0, stop: 2)], int32Arr) coerce Int32 into Float64',
      () {
        NDArray.scope(() {
          final arr = NDArray<Float64>.zeros([4], DType.float64);
          final int32Arr = NDArray<Int32>.fromList(
            <int>[10, 20],
            [2],
            DType.int32,
          );

          arr[const Slice(start: 0, stop: 2)] = int32Arr;
          expect(arr.getCell([0]), closeTo(10.0, 1e-12));
          expect(arr.getCell([1]), closeTo(20.0, 1e-12));
          expect(arr.getCell([2]), closeTo(0.0, 1e-12));
          expect(arr.getCell([3]), closeTo(0.0, 1e-12));

          final int32Arr2 = NDArray<Int32>.fromList(
            <int>[30, 40],
            [2],
            DType.int32,
          );
          arr.sliceAssign([const Slice(start: 2, stop: 4)], int32Arr2);
          expect(arr.getCell([0]), closeTo(10.0, 1e-12));
          expect(arr.getCell([1]), closeTo(20.0, 1e-12));
          expect(arr.getCell([2]), closeTo(30.0, 1e-12));
          expect(arr.getCell([3]), closeTo(40.0, 1e-12));
        });
      },
    );

    test('sliceAssign with DType coercion and broadcasting on 2D array', () {
      NDArray.scope(() {
        final arr = NDArray<Float64>.zeros([3, 3], DType.float64);
        final intRow = NDArray<Int32>.fromList(
          <int>[7, 8, 9],
          [3],
          DType.int32,
        );

        arr.sliceAssign([const Slice(start: 0, stop: 2)], intRow);
        for (var r = 0; r < 2; r++) {
          expect(arr.getCell([r, 0]), closeTo(7.0, 1e-12));
          expect(arr.getCell([r, 1]), closeTo(8.0, 1e-12));
          expect(arr.getCell([r, 2]), closeTo(9.0, 1e-12));
        }
        expect(arr.getCell([2, 0]), closeTo(0.0, 1e-12));
      });
    });

    test(
      'arr[Indices([0, 1])] = row and arr[[[0, 2]]] = row broadcast 1D row of shape [4] to 2 selected rows of [3, 4] array',
      () {
        NDArray.scope(() {
          final arr1 = NDArray<Float64>.zeros([3, 4], DType.float64);
          final row = NDArray<Float64>.fromList(
            <double>[1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );

          arr1[Indices([0, 1])] = row;
          for (var c = 0; c < 4; c++) {
            expect(arr1.getCell([0, c]), closeTo(c + 1.0, 1e-12));
            expect(arr1.getCell([1, c]), closeTo(c + 1.0, 1e-12));
            expect(arr1.getCell([2, c]), closeTo(0.0, 1e-12));
          }

          final arr2 = NDArray<Float64>.zeros([3, 4], DType.float64);
          arr2[[
                [0, 2],
              ]] =
              row;
          for (var c = 0; c < 4; c++) {
            expect(arr2.getCell([0, c]), closeTo(c + 1.0, 1e-12));
            expect(arr2.getCell([1, c]), closeTo(0.0, 1e-12));
            expect(arr2.getCell([2, c]), closeTo(c + 1.0, 1e-12));
          }
        });
      },
    );
  });
}
