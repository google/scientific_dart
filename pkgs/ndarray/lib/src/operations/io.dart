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

// ignore_for_file: non_constant_identifier_names
import 'dart:typed_data';
import 'dart:io';

import 'package:ffi/ffi.dart';

import '../ndarray.dart';
import '../ndarray_extensions_bindings.dart';

import 'dart:ffi' as ffi;

final _descrRegex = RegExp(r'''['"]descr['"]:\s*['"]([^'"]+)['"]''');
final _fortranRegex = RegExp(
  r'''['"]fortran_order['"]:\s*(True|False)''',
  caseSensitive: false,
);
final _shapeRegex = RegExp(r'''['"]shape['"]:\s*\(?([^\)]*)\)?''');

/// Maps a NumPy descriptor string back to an [NDArray] [DType].
DType<dynamic> _descrToDType(String descr) {
  if (descr.contains('>')) {
    throw UnsupportedError('Big-Endian .npy files are not supported yet.');
  }
  // Strip byte-order indicators if any (e.g., '<', '>', '|')
  final clean = descr.replaceAll(RegExp(r'[<>|]'), '');
  switch (clean) {
    case 'f8':
      return DType.float64;
    case 'f4':
      return DType.float32;
    case 'i8':
      return DType.int64;
    case 'i4':
      return DType.int32;
    case 'c16':
      return DType.complex128;
    case 'c8':
      return DType.complex64;
    case 'u1':
      return DType.uint8;
    case 'i2':
      return DType.int16;
    case 'b1':
      return DType.boolean;
    default:
      throw UnsupportedError('Unsupported NumPy data type descriptor: $descr');
  }
}

/// Save an [NDArray] to disk in the standard NumPy binary format (`.npy`).
///
/// This saves the array's shape, datatype, and raw heap memory in a little-endian
/// format which is 100% cross-language compatible with Python's NumPy.
///
/// **Preconditions:**
/// - [filepath] must be a valid, writable path string.
/// - [a] must not be a disposed [NDArray] instance.
///
/// **Throws:**
/// - It is an error if [a] is disposed.
/// - It is an error if the parent directories cannot be created or the file cannot be written to.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ where $N$ is the total number of elements in the array.
/// - Direct block disk operations are performed for contiguous arrays, writing unmanaged C-heap
///   memory directly to the disk as a native byte view.
/// - Non-contiguous strided or transposed views are copied into a contiguous sequence prior to block serialization.
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NumPy NPY Format Specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
/// for details on the binary format.
void save<T>(String filepath, NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot save a disposed NDArray.');
  }
  final file = File(filepath);
  if (!file.parent.existsSync()) {
    file.parent.createSync(recursive: true);
  }

  // Re-orient to contiguous array if a is a strided view to ensure binary file sequentiality
  final NDArray<T> effectiveArray;
  final bool needsDisposeEffective;
  if (!a.isContiguous) {
    effectiveArray = a.copy();
    needsDisposeEffective = true;
  } else {
    effectiveArray = a;
    needsDisposeEffective = false;
  }

  final descr = a.dtype.npyDescriptor;
  final shapeStr = a.shape.length == 1 ? '${a.shape[0]},' : a.shape.join(', ');

  // Build NumPy Version 1.0 Header String dictionary literal
  final headerStr =
      "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";

  // Pad header string so that: magic_string(6) + versions(2) + header_len_bytes(2) + headerStr.length
  // is an exact multiple of 64 bytes.
  final prefixLen = 6 + 2 + 2;
  var paddedHeaderLen =
      ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;

  // Use space padding and end with a mandatory trailing newline '\n'
  final padCount = paddedHeaderLen - headerStr.length - 1;
  final paddedHeader = '$headerStr${' ' * padCount}\n';

  final raf = file.openSync(mode: FileMode.write);

  try {
    // 1. Magic string prefix
    raf.writeFromSync(const [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]); // \x93NUMPY
    // 2. Version 1.0 bytes
    raf.writeFromSync(const [0x01, 0x00]);
    // 3. Little-endian 2-byte unsigned short header length
    final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
    final headerLenBytes = Uint8List(2);
    ByteData.view(headerLenBytes.buffer)
        .setUint16(0, headerBytes.length, Endian.little);
    raf.writeFromSync(headerLenBytes);

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
    if (needsDisposeEffective) {
      effectiveArray.dispose();
    }
  }
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
NDArray<dynamic> load(String filepath) {
  final file = File(filepath);
  if (!file.existsSync()) {
    throw FileSystemException('File not found for load', filepath);
  }

  final raf = file.openSync(mode: FileMode.read);

  try {
    // 1. Check magic prefix
    final magic = raf.readSync(6);
    if (magic.length != 6 ||
        magic[0] != 0x93 ||
        magic[1] != 0x4e ||
        magic[2] != 0x55 ||
        magic[3] != 0x4d ||
        magic[4] != 0x50 ||
        magic[5] != 0x59) {
      throw FormatException('Invalid NumPy .npy binary file signature');
    }

    // 2. Read version
    final version = raf.readSync(2);
    if (version.length != 2) {
      throw FormatException('Failed to read .npy format version headers');
    }

    // 3. Read header length
    final lenBytes = raf.readSync(2);
    final headerLen = ByteData.view(lenBytes.buffer)
        .getUint16(0, Endian.little);

    // 4. Read ASCII Header dictionary
    final headerBytes = raf.readSync(headerLen);
    final headerStr = String.fromCharCodes(headerBytes);

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
    for (var tok in shapeTokens) {
      final cleanTok = tok.trim();
      if (cleanTok.isNotEmpty) {
        shape.add(int.parse(cleanTok));
      }
    }

    // 5. Allocate matching NDArray with target layout strategies
    final elementCount = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);
    final byteSize = elementCount * dtype.byteWidth;

    List<int>? strides;
    // Wire Zero-Copy Column-Major Fortran strides if the file demands it!
    if (fortranOrder && shape.length > 1) {
      final fStrides = List<int>.filled(shape.length, 0);
      var stride = 1;
      for (var i = 0; i < shape.length; i++) {
        fStrides[i] = stride;
        stride *= shape[i];
      }
      strides = fStrides;
    }

    final result = NDArray.create(shape, dtype, strides: strides);

    // 6. Zero-Copy direct stream file read straight into C Heap pointers!
    final byteView = result.pointer.cast<ffi.Uint8>().asTypedList(byteSize);
    raf.readIntoSync(byteView);

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
  Map<String, NDArray<dynamic>> arrays, {
  bool compressed = false,
}) {
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
  final toDispose = <NDArray<dynamic>>[];

  try {
    using((Arena arena) {
      final cNames = arena<ffi.Pointer<ffi.Char>>(numArrays);
      final cHeaderBytes = arena<ffi.Pointer<ffi.Uint8>>(numArrays);
      final cHeaderLens = arena<ffi.Size>(numArrays);
      final cDataPtrs = arena<ffi.Pointer<ffi.Void>>(numArrays);
      final cDataLens = arena<ffi.Size>(numArrays);

      var idx = 0;
      for (final entry in arrays.entries) {
        final arr = entry.value;
        final NDArray<dynamic> effectiveArray;
        if (!arr.isContiguous) {
          effectiveArray = arr.copy();
          toDispose.add(effectiveArray);
        } else {
          effectiveArray = arr;
        }

        final entryName = '${entry.key}.npy';
        cNames[idx] = entryName.toNativeUtf8(allocator: arena).cast<ffi.Char>();

        final descr = effectiveArray.dtype.npyDescriptor;
        final shapeStr = effectiveArray.shape.length == 1
            ? '${effectiveArray.shape[0]},'
            : effectiveArray.shape.join(', ');
        final headerStr =
            "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";

        final prefixLen = 6 + 2 + 2;
        final paddedHeaderLen =
            ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
        final padCount = paddedHeaderLen - headerStr.length - 1;
        final paddedHeader = '$headerStr${' ' * padCount}\n';

        final headerCodeUnits = paddedHeader.codeUnits;
        final totalHeaderBytes = 10 + headerCodeUnits.length;
        final hBuf = arena<ffi.Uint8>(totalHeaderBytes);

        // 1. Magic "\x93NUMPY"
        hBuf[0] = 0x93;
        hBuf[1] = 0x4e; // 'N'
        hBuf[2] = 0x55; // 'U'
        hBuf[3] = 0x4d; // 'M'
        hBuf[4] = 0x50; // 'P'
        hBuf[5] = 0x59; // 'Y'
        // 2. Version 1.0
        hBuf[6] = 0x01;
        hBuf[7] = 0x00;
        // 3. Header length uint16 little-endian
        final hLen = headerCodeUnits.length;
        hBuf[8] = hLen & 0xFF;
        hBuf[9] = (hLen >> 8) & 0xFF;
        // 4. ASCII Header dictionary
        for (var j = 0; j < hLen; j++) {
          hBuf[10 + j] = headerCodeUnits[j];
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

      final cFilepath = filepath
          .toNativeUtf8(allocator: arena)
          .cast<ffi.Char>();
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
    });
  } finally {
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
Map<String, NDArray<dynamic>> loadz(String filepath) {
  final file = File(filepath);
  if (!file.existsSync()) {
    throw FileSystemException('File not found for loadz npz', filepath);
  }

  return using((Arena arena) {
    final cFilepath = filepath.toNativeUtf8(allocator: arena).cast<ffi.Char>();
    final pNumEntries = arena<ffi.Int64>();

    final handle = npz_open_reader(cFilepath, pNumEntries);
    if (handle.address == 0) {
      throw FormatException('Invalid or corrupted .npz ZIP archive: $filepath');
    }

    try {
      final numEntries = pNumEntries.value;
      final results = <String, NDArray<dynamic>>{};

      const nameBufLen = 512;
      final nameBuf = arena<ffi.Char>(nameBufLen);
      const headerBufLen = 65536;
      final headerBuf = arena<ffi.Uint8>(headerBufLen);
      final pHeaderLen = arena<ffi.Size>();
      final pDataLen = arena<ffi.Size>();

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

        if (infoStatus != 0) {
          if (infoStatus == -8) {
            // Corrupted magic bytes in .npy file
            throw FormatException('Invalid .npy magic header in archive entry');
          }
          continue;
        }

        final filename = nameBuf.cast<Utf8>().toDartString();
        if (!filename.endsWith('.npy')) {
          continue;
        }
        final key = filename.substring(0, filename.length - 4);

        final headerLen = pHeaderLen.value;
        final dataLen = pDataLen.value;

        final asciiHeaderLen = headerLen - 10;
        final headerBytes = (headerBuf + 10).asTypedList(asciiHeaderLen);
        final headerStr = String.fromCharCodes(headerBytes);

        final descrMatch = _descrRegex.firstMatch(headerStr);
        if (descrMatch == null) {
          throw FormatException(
            'Invalid npy header: could not parse "descr" parameter string',
          );
        }
        final descr = descrMatch.group(1)!;
        final dtype = _descrToDType(descr);

        final fortMatch = _fortranRegex.firstMatch(headerStr);
        if (fortMatch == null) {
          throw FormatException(
            'Invalid npy header: could not parse "fortran_order" boolean flag',
          );
        }
        final fortranOrder = fortMatch.group(1)!.toLowerCase() == 'true';

        final shapeMatch = _shapeRegex.firstMatch(headerStr);
        if (shapeMatch == null) {
          throw FormatException(
            'Invalid npy header: could not parse "shape" tuple tokens',
          );
        }
        final shapeTokens = shapeMatch.group(1)!.split(',');
        final shape = <int>[];
        for (final tok in shapeTokens) {
          final cleanTok = tok.trim();
          if (cleanTok.isNotEmpty) shape.add(int.parse(cleanTok));
        }

        List<int>? strides;
        if (fortranOrder && shape.length > 1) {
          final fStrides = List<int>.filled(shape.length, 0);
          var stride = 1;
          for (var s = 0; s < shape.length; s++) {
            fStrides[s] = stride;
            stride *= shape[s];
          }
          strides = fStrides;
        }

        final loadedArray = NDArray.create(shape, dtype, strides: strides);

        final extractStatus = npz_reader_extract_data(
          handle,
          i,
          headerLen,
          loadedArray.pointer.cast<ffi.Void>(),
          dataLen,
        );

        if (extractStatus != 0) {
          loadedArray.dispose();
          throw FormatException(
            'Failed to extract .npy array data from .npz entry (index: $i, key: $key, code: $extractStatus)',
          );
        }

        results[key] = loadedArray;
      }

      return results;
    } finally {
      npz_close_reader(handle);
    }
  });
}
