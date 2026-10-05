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
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:meta/meta.dart';
import '../ndarray.dart';
import '../ndarray_extensions_bindings.dart';
import '../scratch_arena.dart';
import 'dart:ffi' as ffi;

ffi.Pointer<ffi.Char> _toNativeUtf8InScratchArena(String s) {
  final units = utf8.encode(s);
  final ptr = ScratchArena.allocate<ffi.Uint8>(units.length + 1);
  ptr.asTypedList(units.length).setAll(0, units);
  ptr[units.length] = 0;
  return ptr.cast<ffi.Char>();
}

final _descrRegex = RegExp(r'''['"]descr['"]:\s*['"]([^'"]+)['"]''');
final _fortranRegex = RegExp(
  r'''['"]fortran_order['"]:\s*(True|False)''',
  caseSensitive: false,
);
final _shapeRegex = RegExp(r'''['"]shape['"]:\s*\(?([^\)]*)\)?''');

/// Maps a NumPy descriptor string back to an [NDArray] [DType].
DType<DTypeTag> _descrToDType(String descr) {
  if (descr.contains('>')) {
    throw UnsupportedError('Big-Endian .npy files are not supported yet.');
  }
  // Strip byte-order indicators if any (e.g., '<', '>', '|', '=')
  final clean = descr.replaceAll(RegExp(r'[<>|=]'), '');
  switch (clean) {
    case 'f8':
      return DType.float64;
    case 'f4':
      return DType.float32;
    case 'f2':
      return DType.float16;
    case 'b1':
      return DType.boolean;
    case 'i8':
      return DType.int64;
    case 'i4':
      return DType.int32;
    case 'i2':
      return DType.int16;
    case 'i1':
      return DType.int8;
    case 'u8':
      return DType.uint64;
    case 'u4':
      return DType.uint32;
    case 'u2':
      return DType.uint16;
    case 'u1':
      return DType.uint8;
    case 'c16':
      return DType.complex128;
    case 'c8':
      return DType.complex64;
    case 'V2' || 'b2' || 'bfloat16':
      return DType.bfloat16;
    default:
      throw UnsupportedError('Unsupported NumPy data type descriptor: $descr');
  }
}

/// Save an [NDArray] to a binary file in NumPy `.npy` format (Version 1.0 or 2.0).
///
/// Serializes array shape, data type, memory ordering, and raw buffer bytes into a standard `.npy` binary stream.
/// Automatically writes Version 1.0 format for standard headers, or Version 2.0 format if header length exceeds 65535 bytes.
///
/// **Preconditions:**
/// - It is an error if [a] is disposed.
/// - [filepath] must be a valid, writable filesystem path.
///
/// **Performance considerations:**
/// - If [a] is C-contiguous (`isContiguous == true`), performs a **zero-copy raw binary write** directly from the native C heap buffer to the file stream in $O(1)$ Dart memory overhead.
/// - If [a] is a non-contiguous view (e.g., sliced or transposed), a temporary contiguous copy is allocated, written, and immediately disposed.
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NumPy save reference](https://numpy.org/doc/stable/reference/generated/numpy.save.html)
/// and [NPY format specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) for details.
void save<T extends DTypeTag>(String filepath, NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot save a disposed NDArray.');
  }
  final file = File(filepath);
  if (!file.parent.existsSync()) {
    file.parent.createSync(recursive: true);
  }

  // Re-orient to contiguous array if a is a strided view to ensure binary file sequentiality
  final effectiveArray = a.isContiguous ? a : a.copy();
  try {
    final descr = a.dtype.npyDescriptor;
    final shapeStr = a.shape.length == 1
        ? '${a.shape[0]},'
        : a.shape.join(', ');

    // Build NumPy Version 1.0 Header String dictionary literal
    final headerStr =
        "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";

    // Format version: 1.0 (2-byte header length, 10-byte prefix) or 2.0 (4-byte header length, 12-byte prefix)
    // If header length exceeds 65535 bytes, version 2.0 is required.
    // First calculate using v1.0 prefixLen (10); if paddedHeaderLen > 65535, recalculate with v2.0 prefixLen (12).
    var prefixLen = 6 + 2 + 2;
    var paddedHeaderLen =
        ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
    final isVersion2 = paddedHeaderLen > 65535;
    if (isVersion2) {
      prefixLen = 6 + 2 + 4;
      paddedHeaderLen =
          ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
    }

    // Use space padding and end with a mandatory trailing newline '\n'
    final padCount = paddedHeaderLen - headerStr.length - 1;
    final paddedHeader = '$headerStr${' ' * padCount}\n';

    final raf = file.openSync(mode: FileMode.write);

    try {
      // 1. Magic string prefix
      raf.writeFromSync(const [
        0x93,
        0x4e,
        0x55,
        0x4d,
        0x50,
        0x59,
      ]); // \x93NUMPY
      final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
      if (headerBytes.length > 65535 || isVersion2) {
        // Version 2.0 bytes
        raf.writeFromSync(const [0x02, 0x00]);
        // Little-endian 4-byte unsigned int header length
        final headerLenBytes = Uint8List(4);
        ByteData.view(
          headerLenBytes.buffer,
        ).setUint32(0, headerBytes.length, Endian.little);
        raf.writeFromSync(headerLenBytes);
      } else {
        // Version 1.0 bytes
        raf.writeFromSync(const [0x01, 0x00]);
        // Little-endian 2-byte unsigned short header length
        final headerLenBytes = Uint8List(2);
        ByteData.view(
          headerLenBytes.buffer,
        ).setUint16(0, headerBytes.length, Endian.little);
        raf.writeFromSync(headerLenBytes);
      }

      // 4. Write ASCII header dictionary
      raf.writeFromSync(headerBytes);

      // 5. Zero-Copy Raw C-Heap Bytes block dump!
      final elementCount = effectiveArray.shape.isEmpty
          ? 1
          : effectiveArray.shape.reduce((x, y) => x * y);
      final byteSize = elementCount * effectiveArray.dtype.byteWidth;
      final byteView = effectiveArray.pointer.cast<ffi.Uint8>().asTypedList(
        byteSize,
      );
      raf.writeFromSync(byteView);
    } finally {
      raf.closeSync();
    }
  } finally {
    if (!identical(effectiveArray, a)) {
      effectiveArray.dispose();
    }
  }
}

Uint8List _readExactSync(RandomAccessFile raf, int count) {
  final buffer = Uint8List(count);
  var totalRead = 0;
  while (totalRead < count) {
    final n = raf.readIntoSync(buffer, totalRead, count);
    if (n <= 0) break;
    totalRead += n;
  }
  if (totalRead != count) {
    throw FormatException(
      'Unexpected EOF while reading NPY stream: expected $count bytes, got $totalRead',
    );
  }
  return buffer;
}

/// Internal header information parsed from a NumPy `.npy` header dictionary.
///
/// Throws a [FormatException] if the header string lacks required fields
/// (`descr`, `fortran_order`, `shape`) or contains negative dimensions.
@internal
({DType<DTypeTag> dtype, bool fortranOrder, List<int> shape}) parseNpyHeader(
  String headerStr,
) {
  // Parse descr via regex
  final descrMatch = _descrRegex.firstMatch(headerStr);
  if (descrMatch == null) {
    throw FormatException(
      'Invalid npy header: could not parse "descr" parameter string',
    );
  }
  final descr = descrMatch.group(1)!;
  final dtype = _descrToDType(descr);

  // Parse fortran_order bool flag
  final fortMatch = _fortranRegex.firstMatch(headerStr);
  if (fortMatch == null) {
    throw FormatException(
      'Invalid npy header: could not parse "fortran_order" boolean flag',
    );
  }
  final fortranOrder = fortMatch.group(1)!.toLowerCase() == 'true';

  // Parse shape tuple
  final shapeMatch = _shapeRegex.firstMatch(headerStr);
  if (shapeMatch == null) {
    throw FormatException(
      'Invalid npy header: could not parse "shape" tuple tokens',
    );
  }
  final shapeTokens = shapeMatch.group(1)!.split(',');
  final shape = <int>[];
  for (final tok in shapeTokens) {
    var cleanTok = tok.trim();
    if (cleanTok.isNotEmpty) {
      if (cleanTok.endsWith('L') || cleanTok.endsWith('l')) {
        cleanTok = cleanTok.substring(0, cleanTok.length - 1).trim();
      }
      final dim = int.parse(cleanTok);
      if (dim < 0) {
        throw FormatException('Shape dimensions cannot be negative: $dim');
      }
      shape.add(dim);
    }
  }

  return (dtype: dtype, fortranOrder: fortranOrder, shape: shape);
}

/// Load an [NDArray] binary data block from a NumPy `.npy` file.
///
/// **Preconditions:**
/// - [filepath] must point to an existing, readable file on the file system.
///
/// **Throws:**
/// - It is an error if the file does not exist or cannot be read.
/// - It is an error if the file lacks a valid `.npy` magic prefix, version header,
///   or contains corrupted ASCII headers or shapes.
/// - It is an error if the file uses Big-Endian byte order or contains an unsupported
///   NumPy data type descriptor (e.g., f16, u2, etc.).
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ where $N$ is the total number of elements in the loaded array.
/// - Performs direct binary block transfers from the file stream into unmanaged C-heap memory pages.
/// - Supports **Column-Major Fortran strides mapping** in $O(1)$ time: if a file is flagged
///   as `fortran_order: True` (column-major from Python), it loads sequential columns into the C heap
///   and configures column-major strides directly, without reshaping data.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. **The caller takes full ownership** of this memory and **must explicitly call [dispose]** to prevent native leaks, unless executing inside a managed [NDArray.scope].
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NumPy NPY Format Specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
/// for details on the binary format.
NDArray<DTypeTag> load(String filepath) {
  final file = File(filepath);
  if (!file.existsSync()) {
    throw FileSystemException('File not found for load', filepath);
  }

  final raf = file.openSync(mode: FileMode.read);

  try {
    // 1. Check magic prefix
    final magic = _readExactSync(raf, 6);
    if (magic[0] != 0x93 ||
        magic[1] != 0x4e ||
        magic[2] != 0x55 ||
        magic[3] != 0x4d ||
        magic[4] != 0x50 ||
        magic[5] != 0x59) {
      throw FormatException('Invalid NumPy .npy binary file signature');
    }

    // 2. Read version
    final version = _readExactSync(raf, 2);
    final majorVersion = version[0];
    final minorVersion = version[1];
    if (majorVersion != 1 && majorVersion != 2 && majorVersion != 3) {
      throw FormatException(
        'Unsupported .npy format version: $majorVersion.$minorVersion',
      );
    }

    // 3. Read header length
    const maxNpyHeaderBytes = 1024 * 1024;
    final int headerLen;
    if (majorVersion >= 2) {
      final lenBytes = _readExactSync(raf, 4);
      headerLen = ByteData.sublistView(lenBytes).getUint32(0, Endian.little);
    } else {
      final lenBytes = _readExactSync(raf, 2);
      headerLen = ByteData.sublistView(lenBytes).getUint16(0, Endian.little);
    }
    if (headerLen > maxNpyHeaderBytes) {
      throw FormatException(
        'NPY header length ($headerLen) exceeds maximum allowed size ($maxNpyHeaderBytes bytes).',
      );
    }

    // 4. Read ASCII/UTF-8 Header dictionary
    final headerBytes = _readExactSync(raf, headerLen);
    final headerStr = utf8.decode(headerBytes, allowMalformed: true);

    final (:dtype, :fortranOrder, :shape) = parseNpyHeader(headerStr);

    // 5. Allocate matching NDArray with target layout strategies
    final elementCount = checkTotalSize(shape);
    if (elementCount > 0x7fffffffffffffff ~/ dtype.byteWidth) {
      throw ArgumentError.value(
        shape,
        'shape',
        'Must not result in byte size overflowing 64-bit integer',
      );
    }
    final byteSize = elementCount * dtype.byteWidth;
    final remainingBytes = raf.lengthSync() - raf.positionSync();
    if (remainingBytes < byteSize) {
      throw FormatException(
        'Unexpected EOF while reading NPY payload: expected $byteSize bytes, got $remainingBytes',
      );
    }

    List<int>? strides;
    // Wire Zero-Copy Column-Major Fortran strides if the file demands it!
    if (fortranOrder && shape.length > 1) {
      final fStrides = List<int>.filled(shape.length, 0);
      var stride = 1;
      for (var i = 0; i < shape.length; i++) {
        fStrides[i] = stride;
        final dim = shape[i];
        if (stride != 0 && dim > 0 && stride > 0x7fffffffffffffff ~/ dim) {
          throw ArgumentError.value(
            shape,
            'shape',
            'Must not result in Fortran-contiguous stride overflowing 64-bit integer',
          );
        }
        stride *= dim;
      }
      strides = fStrides;
    }

    final result = NDArray.create(shape, dtype, strides: strides);

    // 6. Zero-Copy direct stream file read straight into C Heap pointers!
    try {
      final byteView = result.pointer.cast<ffi.Uint8>().asTypedList(byteSize);
      var totalRead = 0;
      while (totalRead < byteSize) {
        final bytesRead = raf.readIntoSync(byteView, totalRead, byteSize);
        if (bytesRead <= 0) {
          break;
        }
        totalRead += bytesRead;
      }
      if (totalRead != byteSize) {
        throw FormatException(
          'Unexpected EOF while reading NPY payload: expected $byteSize bytes, got $totalRead',
        );
      }
    } catch (_) {
      result.dispose();
      rethrow;
    }

    return result;
  } finally {
    raf.closeSync();
  }
}

/// Save multiple named arrays to a single ZIP archive file on disk (`.npz`).
///
/// This corresponds to NumPy's `savez` and `savez_compressed` functions.
///
/// Uses high-performance native C zip streaming to serialize array headers and raw memory
/// directly to disk without intermediate memory buffer copies.
///
/// **Preconditions:**
/// - [filepath] must be a valid, writable path string.
/// - [arrays] map must not be empty, and all [NDArray] values must not be disposed.
///
/// **Errors & Exceptions:**
/// - It is an error if any array in [arrays] is disposed.
/// - Throws [FormatException] if the native ZIP archive encoding fails.
/// - Throws [FileSystemException] if the parent directories cannot be created or the archive file cannot be written.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ where $N$ is the total number of elements across all packed arrays.
/// - Performs zero-copy streaming: reads unmanaged C-heap memory pages directly in chunks into the native
///   ZIP writer without allocating transient Dart heap arrays.
/// - If [compressed] is true, applies native Deflate compression.
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NumPy savez reference](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
/// for details on NumPy archive formats.
void savez(
  String filepath,
  Map<String, NDArray<DTypeTag>> arrays, {
  bool compressed = false,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty');
  }
  for (final entry in arrays.entries) {
    if (entry.value.isDisposed) {
      throw StateError('Cannot save a disposed NDArray (key: ${entry.key}).');
    }
  }

  final file = File(filepath);
  if (!file.parent.existsSync()) {
    file.parent.createSync(recursive: true);
  }

  final numArrays = arrays.length;
  final toDispose = <NDArray<DTypeTag>>[];

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

    var idx = 0;
    for (final entry in arrays.entries) {
      final arr = entry.value;
      final NDArray<DTypeTag> effectiveArray;
      if (!arr.isContiguous) {
        effectiveArray = arr.copy();
        toDispose.add(effectiveArray);
      } else {
        effectiveArray = arr;
      }

      final entryName = '${entry.key}.npy';
      cNames[idx] = _toNativeUtf8InScratchArena(entryName);

      final descr = effectiveArray.dtype.npyDescriptor;
      final shapeStr = effectiveArray.shape.length == 1
          ? '${effectiveArray.shape[0]},'
          : effectiveArray.shape.join(', ');
      final headerStr =
          "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";

      var prefixLen = 6 + 2 + 2;
      var paddedHeaderLen =
          ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
      final isVersion2 = paddedHeaderLen > 65535;
      if (isVersion2) {
        prefixLen = 6 + 2 + 4;
        paddedHeaderLen =
            ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
      }
      final padCount = paddedHeaderLen - headerStr.length - 1;
      final paddedHeader = '$headerStr${' ' * padCount}\n';

      final headerCodeUnits = paddedHeader.codeUnits;
      final hLen = headerCodeUnits.length;
      final totalHeaderBytes = prefixLen + hLen;
      final hBuf = ScratchArena.allocate<ffi.Uint8>(totalHeaderBytes);

      // 1. Magic "\x93NUMPY"
      hBuf[0] = 0x93;
      hBuf[1] = 0x4e; // 'N'
      hBuf[2] = 0x55; // 'U'
      hBuf[3] = 0x4d; // 'M'
      hBuf[4] = 0x50; // 'P'
      hBuf[5] = 0x59; // 'Y'
      if (isVersion2 || hLen > 65535) {
        // 2. Version 2.0
        hBuf[6] = 0x02;
        hBuf[7] = 0x00;
        // 3. Header length uint32 little-endian
        hBuf[8] = hLen & 0xFF;
        hBuf[9] = (hLen >> 8) & 0xFF;
        hBuf[10] = (hLen >> 16) & 0xFF;
        hBuf[11] = (hLen >> 24) & 0xFF;
        // 4. ASCII Header dictionary
        for (var j = 0; j < hLen; j++) {
          hBuf[12 + j] = headerCodeUnits[j];
        }
      } else {
        // 2. Version 1.0
        hBuf[6] = 0x01;
        hBuf[7] = 0x00;
        // 3. Header length uint16 little-endian
        hBuf[8] = hLen & 0xFF;
        hBuf[9] = (hLen >> 8) & 0xFF;
        // 4. ASCII Header dictionary
        for (var j = 0; j < hLen; j++) {
          hBuf[10 + j] = headerCodeUnits[j];
        }
      }

      cHeaderBytes[idx] = hBuf;
      cHeaderLens[idx] = totalHeaderBytes;

      final elementCount = effectiveArray.shape.isEmpty
          ? 1
          : effectiveArray.shape.reduce((x, y) => x * y);
      final byteSize = elementCount * effectiveArray.dtype.byteWidth;

      cDataPtrs[idx] = effectiveArray.pointer.cast<ffi.Void>();
      cDataLens[idx] = byteSize;
      idx++;
    }

    final cFilepath = _toNativeUtf8InScratchArena(filepath);
    final compressLevel = compressed ? 6 : 0;

    final status = npz_save(
      cFilepath,
      numArrays,
      cNames,
      cHeaderBytes,
      cHeaderLens,
      cDataPtrs,
      cDataLens,
      compressLevel,
    );

    if (status != 0) {
      throw FormatException(
        'Failed to encode .npz zip archive format bytes (error code: $status)',
      );
    }
  } finally {
    ScratchArena.reset(marker);
    for (final a in toDispose) {
      a.dispose();
    }
  }
}

/// Load multiple named [NDArray] instances back from a NumPy `.npz` ZIP archive.
///
/// Unpacks and deserializes all inner `.npy` files, mapping variable name keys to loaded array targets.
/// Supports compressed and uncompressed Python-generated `.npz` archives with zero memory copying.
///
/// **Preconditions:**
/// - [filepath] must point to an existing, readable `.npz` ZIP archive file on the filesystem.
///
/// **Errors & Exceptions:**
/// - Throws [FileSystemException] if the file does not exist or cannot be read.
/// - Throws [FormatException] if the file is not a valid ZIP archive or contains corrupted inner `.npy` byte streams.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ where $N$ is the total number of elements across all packed arrays.
/// - **Zero-copy Native Decompression:** Reads/decompresses stream data directly from native zip file
///   iterators straight into unmanaged C-heap array memory without intermediate Dart heap byte lists.
/// - Supports **Column-Major Fortran strides mapping** in $O(1)$ time for column-major `.npy` entries.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates new arrays on the unmanaged C heap. **The caller takes full ownership** of this memory and **must explicitly call [dispose]** on all returned arrays in the map to prevent native leaks, unless executing inside a managed [NDArray.scope].
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NumPy load reference](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
/// and [ZIP format details](https://en.wikipedia.org/wiki/ZIP_(file_format)) for additional information.
Map<String, NDArray<DTypeTag>> loadz(String filepath) {
  final file = File(filepath);
  if (!file.existsSync()) {
    throw FileSystemException('File not found for loadz npz', filepath);
  }

  final marker = ScratchArena.marker;
  try {
    final cFilepath = _toNativeUtf8InScratchArena(filepath);
    final pNumEntries = ScratchArena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>(),
    );

    final handle = npz_open_reader(cFilepath, pNumEntries);
    if (handle.address == 0) {
      throw FormatException('Invalid or corrupted .npz ZIP archive: $filepath');
    }

    try {
      final numEntries = pNumEntries.value;
      final results = <String, NDArray<DTypeTag>>{};

      const nameBufLen = 4096;
      final nameBuf = ScratchArena.allocate<ffi.Char>(
        nameBufLen * ffi.sizeOf<ffi.Char>(),
      );
      const headerBufLen = 65536;
      final headerBuf = ScratchArena.allocate<ffi.Uint8>(headerBufLen);
      final pHeaderLen = ScratchArena.allocate<ffi.Size>(
        ffi.sizeOf<ffi.Size>(),
      );
      final pDataLen = ScratchArena.allocate<ffi.Size>(ffi.sizeOf<ffi.Size>());

      try {
        for (var i = 0; i < numEntries; i++) {
          final infoStatus = npz_reader_get_entry_info(
            handle,
            i,
            nameBuf,
            nameBufLen,
            headerBuf,
            headerBufLen,
            pHeaderLen,
            pDataLen,
          );

          ffi.Pointer<ffi.Uint8> activeHeaderBuf = headerBuf;
          if (infoStatus != 0) {
            if (infoStatus == -3 || infoStatus == -4) {
              continue;
            }
            if (infoStatus == -9) {
              const maxNpyHeaderBytes = 1024 * 1024;
              final requiredHeaderLen = pHeaderLen.value;
              if (requiredHeaderLen > maxNpyHeaderBytes) {
                throw FormatException(
                  'NPY header length ($requiredHeaderLen) exceeds maximum allowed size ($maxNpyHeaderBytes bytes).',
                );
              }
              final largeHeaderBuf = ScratchArena.allocate<ffi.Uint8>(
                requiredHeaderLen,
              );
              final retryStatus = npz_reader_get_entry_info(
                handle,
                i,
                nameBuf,
                nameBufLen,
                largeHeaderBuf,
                requiredHeaderLen,
                pHeaderLen,
                pDataLen,
              );
              if (retryStatus != 0) {
                throw FormatException(
                  'Corrupted or invalid .npy entry in .npz archive (status: $retryStatus)',
                );
              }
              activeHeaderBuf = largeHeaderBuf;
            } else if (infoStatus == -8) {
              // Corrupted magic bytes in .npy file
              throw FormatException(
                'Invalid .npy magic header in archive entry',
              );
            } else {
              throw FormatException(
                'Corrupted or invalid .npy entry in .npz archive (status: $infoStatus)',
              );
            }
          }

          final filename = nameBuf.cast<Utf8>().toDartString();
          if (!filename.endsWith('.npy')) {
            continue;
          }
          final key = filename.substring(0, filename.length - 4);

          final realHeaderLen = pHeaderLen.value;
          final realDataLen = pDataLen.value;
          final majorVersion = activeHeaderBuf[6];
          final minorVersion = activeHeaderBuf[7];
          if (majorVersion != 1 && majorVersion != 2 && majorVersion != 3) {
            throw FormatException(
              'Unsupported .npy format version: $majorVersion.$minorVersion',
            );
          }
          final prefixLen = majorVersion >= 2 ? 12 : 10;
          final asciiLen = math.max(0, realHeaderLen - prefixLen);
          final headerBytes = (activeHeaderBuf + prefixLen).asTypedList(
            asciiLen,
          );
          final headerStr = utf8.decode(headerBytes, allowMalformed: true);

          final (:dtype, :fortranOrder, :shape) = parseNpyHeader(headerStr);

          List<int>? strides;
          if (fortranOrder && shape.length > 1) {
            final fStrides = List<int>.filled(shape.length, 0);
            var stride = 1;
            for (var s = 0; s < shape.length; s++) {
              fStrides[s] = stride;
              final dim = shape[s];
              if (stride != 0 &&
                  dim > 0 &&
                  stride > 0x7fffffffffffffff ~/ dim) {
                throw ArgumentError.value(
                  shape,
                  'shape',
                  'Must not result in Fortran-contiguous stride overflowing 64-bit integer',
                );
              }
              stride *= dim;
            }
            strides = fStrides;
          }

          final totalElements = checkTotalSize(shape);
          if (totalElements > 0x7fffffffffffffff ~/ dtype.byteWidth) {
            throw ArgumentError.value(
              shape,
              'shape',
              'Must not result in byte size overflowing 64-bit integer',
            );
          }
          final expectedBytes = totalElements * dtype.byteWidth;
          if (realDataLen != expectedBytes) {
            throw FormatException(
              'Mismatched data size in NPZ archive for entry: expected $expectedBytes bytes from NPY header, got $realDataLen bytes from ZIP header.',
            );
          }

          final loadedArray = NDArray.create(shape, dtype, strides: strides);

          final extractStatus = npz_reader_extract_data(
            handle,
            i,
            realHeaderLen,
            loadedArray.pointer.cast<ffi.Void>(),
            expectedBytes,
            realDataLen,
          );

          if (extractStatus != 0) {
            loadedArray.dispose();
            throw FormatException(
              'Failed to extract .npy array data from .npz entry (index: $i, key: $key, code: $extractStatus)',
            );
          }

          results[key]?.dispose();
          results[key] = loadedArray;
        }

        return results;
      } catch (e) {
        for (final arr in results.values) {
          arr.dispose();
        }
        rethrow;
      }
    } finally {
      npz_close_reader(handle);
    }
  } finally {
    ScratchArena.reset(marker);
  }
}
