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
import 'dart:io' show FileSystemException, OSError;
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' show malloc;
import 'package:meta/meta.dart';
import '../ndarray.dart';
import '../ndarray_extensions_bindings.dart';
import '../scratch_arena.dart';
import '../wasm_pointer_lists.dart'
    show copyBytesToNative, copyNativeToBytes, maxWasm32AllocationBytes;
import 'helpers.dart' show checkNativeOom;

ffi.Pointer<ffi.Char> _toNativeUtf8InScratchArena(String s) {
  final units = utf8.encode(s);
  final ptr = ScratchArena.allocate<ffi.Uint8>(units.length + 1);
  for (var i = 0; i < units.length; i++) {
    ptr[i] = units[i];
  }
  ptr[units.length] = 0;
  return ptr.cast<ffi.Char>();
}

String _fromNativeUtf8(ffi.Pointer<ffi.Char> ptr) {
  final u8 = ptr.cast<ffi.Uint8>();
  var len = 0;
  while (u8[len] != 0) {
    len++;
  }
  final units = Uint8List(len);
  for (var i = 0; i < len; i++) {
    units[i] = u8[i];
  }
  return utf8.decode(units, allowMalformed: true);
}

/// Builds a [FileSystemException] for [filepath] whose [OSError] carries the
/// `errno` and message of the most recent failed native file operation.
///
/// Must be called synchronously after the failing native call, before any
/// other file operation on this isolate.
FileSystemException _nativeFileException(String message, String filepath) {
  final marker = ScratchArena.marker;
  try {
    const capacity = 256;
    final buffer = ScratchArena.allocate<ffi.Uint8>(capacity);
    final errorCode = native_file_last_error(buffer, capacity);
    final osMessage = _fromNativeUtf8(buffer.cast<ffi.Char>());
    return FileSystemException(
      message,
      filepath,
      OSError(osMessage, errorCode),
    );
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Reads exactly [count] bytes at byte [offset] of the open file [handle]
/// into [dest], using [pOutRead] as the native out-parameter.
///
/// Throws a [FormatException] if the file ends before [count] bytes were
/// read, and a [FileSystemException] if the read itself fails.
void _readExact(
  ffi.Pointer<ffi.Void> handle,
  String filepath,
  int offset,
  int count,
  ffi.Pointer<ffi.Void> dest,
  ffi.Pointer<ffi.Int64> pOutRead,
) {
  final status = native_file_handle_read(handle, offset, count, dest, pOutRead);
  if (status != 0) {
    throw _nativeFileException('Failed to read .npy file', filepath);
  }
  if (pOutRead.value != count) {
    throw FormatException(
      'Unexpected EOF while reading NPY stream: expected $count bytes, got ${pOutRead.value}',
    );
  }
}

void _writePointerAt<T extends ffi.NativeType>(
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

void _writeSizeAt(ffi.Pointer<ffi.Size> array, int index, int value) {
  if (ffi.sizeOf<ffi.Size>() == 4) {
    array.cast<ffi.Uint32>()[index] = value;
  } else {
    array.cast<ffi.Uint64>()[index] = value;
  }
}

int _readSize(ffi.Pointer<ffi.Size> ptr) {
  if (ffi.sizeOf<ffi.Size>() == 4) {
    return ptr.cast<ffi.Uint32>().value;
  } else {
    return ptr.cast<ffi.Uint64>().value;
  }
}

ffi.Pointer<T> _readPointer<T extends ffi.NativeType>(
  ffi.Pointer<ffi.Pointer<T>> ptr,
) {
  final address = ffi.sizeOf<ffi.IntPtr>() == 4
      ? ptr.cast<ffi.Uint32>().value
      : ptr.cast<ffi.Uint64>().value;
  return ffi.Pointer<T>.fromAddress(address);
}

const int _maxNpyHeaderBytes = 1024 * 1024;

/// Encodes a complete 64-byte-aligned NumPy `.npy` header (magic, version
/// 1.0 or 2.0, little-endian length prefix, and space-padded ASCII header
/// dictionary ending in `\n`).
Uint8List _encodeNpyHeader(DType<DTypeTag> dtype, List<int> shape) {
  final descr = dtype.npyDescriptor;
  final shapeStr = shape.length == 1 ? '${shape[0]},' : shape.join(', ');
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
  final useV2 = hLen > 65535 || isVersion2;
  final actualPrefixLen = useV2 ? 12 : 10;
  final out = Uint8List(actualPrefixLen + hLen);

  out[0] = 0x93;
  out[1] = 0x4e; // 'N'
  out[2] = 0x55; // 'U'
  out[3] = 0x4d; // 'M'
  out[4] = 0x50; // 'P'
  out[5] = 0x59; // 'Y'
  if (useV2) {
    out[6] = 0x02;
    out[7] = 0x00;
    out[8] = hLen & 0xFF;
    out[9] = (hLen >> 8) & 0xFF;
    out[10] = (hLen >> 16) & 0xFF;
    out[11] = (hLen >> 24) & 0xFF;
  } else {
    out[6] = 0x01;
    out[7] = 0x00;
    out[8] = hLen & 0xFF;
    out[9] = (hLen >> 8) & 0xFF;
  }
  for (var j = 0; j < hLen; j++) {
    out[actualPrefixLen + j] = headerCodeUnits[j];
  }
  return out;
}

({int prefixLen, int headerLen}) _parseNpyPrefix(
  int availablePrefix,
  int Function(int index) readByte,
) {
  void requirePrefix(int needed) {
    if (availablePrefix < needed) {
      throw FormatException(
        'Unexpected EOF while reading NPY stream: expected $needed bytes, got $availablePrefix',
      );
    }
  }

  requirePrefix(6);
  if (readByte(0) != 0x93 ||
      readByte(1) != 0x4e ||
      readByte(2) != 0x55 ||
      readByte(3) != 0x4d ||
      readByte(4) != 0x50 ||
      readByte(5) != 0x59) {
    throw FormatException('Invalid NumPy .npy binary file signature');
  }

  requirePrefix(8);
  final majorVersion = readByte(6);
  final minorVersion = readByte(7);
  if (majorVersion != 1 && majorVersion != 2 && majorVersion != 3) {
    throw FormatException(
      'Unsupported .npy format version: $majorVersion.$minorVersion',
    );
  }

  final int headerLen;
  final int prefixLen;
  if (majorVersion >= 2) {
    prefixLen = 12;
    requirePrefix(prefixLen);
    headerLen =
        readByte(8) |
        (readByte(9) << 8) |
        (readByte(10) << 16) |
        (readByte(11) << 24);
  } else {
    prefixLen = 10;
    requirePrefix(prefixLen);
    headerLen = readByte(8) | (readByte(9) << 8);
  }
  if (headerLen < 0 || headerLen > _maxNpyHeaderBytes) {
    throw FormatException(
      'NPY header length ($headerLen) exceeds maximum allowed size ($_maxNpyHeaderBytes bytes).',
    );
  }
  return (prefixLen: prefixLen, headerLen: headerLen);
}

({DType<DTypeTag> dtype, List<int> shape, List<int>? strides, int byteSize})
_prepareNpyLayout(String headerStr) {
  final (:dtype, :fortranOrder, :shape) = parseNpyHeader(headerStr);
  final elementCount = checkTotalSize(shape);
  if (elementCount > 0x7fffffffffffffff ~/ dtype.byteWidth) {
    throw ArgumentError.value(
      shape,
      'shape',
      'Must not result in byte size overflowing 64-bit integer',
    );
  }
  final byteSize = elementCount * dtype.byteWidth;

  List<int>? strides;
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

  return (dtype: dtype, shape: shape, strides: strides, byteSize: byteSize);
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

  // Re-orient to contiguous array if a is a strided view to ensure binary file sequentiality
  final effectiveArray = a.isContiguous ? a : a.copy();
  try {
    final headerBytes = _encodeNpyHeader(
      effectiveArray.dtype,
      effectiveArray.shape,
    );
    final marker = ScratchArena.marker;
    try {
      final hBuf = ScratchArena.allocate<ffi.Uint8>(headerBytes.length);
      copyBytesToNative(hBuf.cast<ffi.Void>(), headerBytes, headerBytes.length);

      final elementCount = effectiveArray.shape.isEmpty
          ? 1
          : effectiveArray.shape.reduce((x, y) => x * y);
      final byteSize = elementCount * effectiveArray.dtype.byteWidth;
      final cFilepath = _toNativeUtf8InScratchArena(filepath);
      final status = native_file_write_all(
        cFilepath,
        hBuf,
        headerBytes.length,
        effectiveArray.pointer.cast<ffi.Void>(),
        byteSize,
      );
      if (status != 0) {
        throw _nativeFileException(
          status == -1
              ? 'Cannot open .npy file for writing'
              : 'Failed to write .npy file',
          filepath,
        );
      }
    } finally {
      ScratchArena.reset(marker);
    }
  } finally {
    if (!identical(effectiveArray, a)) {
      effectiveArray.dispose();
    }
  }
}

/// Serializes an [NDArray] into an in-memory NumPy `.npy` binary buffer.
///
/// Produces the same 64-byte-aligned NPY v1.0/v2.0 binary format written by
/// [save], without touching the filesystem — suitable for Web/Wasm runtimes,
/// network transport, or custom storage backends.
///
/// **Preconditions:**
/// - [a] must not be disposed.
///
/// **Performance considerations:**
/// - $O(N)$ memory copy where $N$ is the total number of elements in [a].
/// - Non-contiguous strided views are materialized into a temporary
///   C-contiguous copy before encoding.
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NPY format specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
/// for details on the binary layout.
Uint8List saveBytes<T extends DTypeTag>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot save a disposed NDArray.');
  }

  final effectiveArray = a.isContiguous ? a : a.copy();
  try {
    final headerBytes = _encodeNpyHeader(
      effectiveArray.dtype,
      effectiveArray.shape,
    );
    final elementCount = effectiveArray.shape.isEmpty
        ? 1
        : effectiveArray.shape.reduce((x, y) => x * y);
    final byteSize = elementCount * effectiveArray.dtype.byteWidth;
    final out = Uint8List(headerBytes.length + byteSize);
    for (var i = 0; i < headerBytes.length; i++) {
      out[i] = headerBytes[i];
    }
    if (byteSize > 0) {
      copyNativeToBytes(
        Uint8List.sublistView(out, headerBytes.length),
        effectiveArray.pointer.cast<ffi.Void>(),
        byteSize,
      );
    }
    return out;
  } finally {
    if (!identical(effectiveArray, a)) {
      effectiveArray.dispose();
    }
  }
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
  final marker = ScratchArena.marker;
  try {
    final cFilepath = _toNativeUtf8InScratchArena(filepath);
    final handle = native_file_open_read(cFilepath);
    if (handle.address == 0) {
      throw _nativeFileException('Cannot open .npy file for reading', filepath);
    }
    try {
      final fileSize = native_file_handle_size(handle);
      if (fileSize < 0) {
        throw _nativeFileException(
          'Cannot determine size of .npy file',
          filepath,
        );
      }
      final pOutRead = ScratchArena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>(),
      );

      // 1. Fixed-size prefix: magic (6), version (2), header length (2 for
      // format 1.0, 4 for 2.0 and 3.0). Read as much of it as the file has.
      const maxPrefixLen = 12;
      final prefixBuf = ScratchArena.allocate<ffi.Uint8>(maxPrefixLen);
      final availablePrefix = math.min(maxPrefixLen, fileSize);
      _readExact(
        handle,
        filepath,
        0,
        availablePrefix,
        prefixBuf.cast<ffi.Void>(),
        pOutRead,
      );
      final (:prefixLen, :headerLen) = _parseNpyPrefix(
        availablePrefix,
        (i) => prefixBuf[i],
      );
      final dataOffset = prefixLen + headerLen;

      // 2. ASCII/UTF-8 header dictionary.
      final headerBuf = ScratchArena.allocate<ffi.Uint8>(
        math.max(1, headerLen),
      );
      _readExact(
        handle,
        filepath,
        prefixLen,
        headerLen,
        headerBuf.cast<ffi.Void>(),
        pOutRead,
      );
      final headerBytes = Uint8List(headerLen);
      copyNativeToBytes(headerBytes, headerBuf.cast<ffi.Void>(), headerLen);
      final headerStr = utf8.decode(headerBytes, allowMalformed: true);

      // 3. Allocate matching NDArray with target layout strategies.
      final (:dtype, :shape, :strides, :byteSize) = _prepareNpyLayout(
        headerStr,
      );
      final remainingBytes = fileSize - dataOffset;
      if (remainingBytes < byteSize) {
        throw FormatException(
          'Unexpected EOF while reading NPY payload: expected $byteSize bytes, got $remainingBytes',
        );
      }

      final result = NDArray.create(shape, dtype, strides: strides);

      // 4. Payload straight into the array's native buffer.
      try {
        _readExact(
          handle,
          filepath,
          dataOffset,
          byteSize,
          result.pointer.cast<ffi.Void>(),
          pOutRead,
        );
      } catch (_) {
        result.dispose();
        rethrow;
      }

      return result;
    } finally {
      native_file_close(handle);
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Deserializes an [NDArray] from an in-memory NumPy `.npy` binary buffer.
///
/// Supports NPY v1.0, v2.0, and v3.0 headers, all 15 little-endian [DType]s,
/// non-zero [Uint8List.sublistView] offsets, and zero-copy Column-Major Fortran
/// stride mapping (`fortran_order: True`), without touching the filesystem.
///
/// **Errors & Exceptions:**
/// - Throws a [FormatException] if [bytes] lacks a valid `.npy` magic prefix,
///   uses an unsupported format version, has a malformed or oversized header,
///   or is truncated before the expected payload length.
/// - Throws an [UnsupportedError] if the `.npy` header specifies Big-Endian
///   byte order or an unsupported NumPy dtype descriptor.
///
/// **Performance considerations:**
/// - $O(N)$ memory copy where $N$ is the total number of elements in the array.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes ownership
///   and must call [NDArray.dispose] unless running inside an [NDArray.scope].
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NPY format specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
/// for details on the binary layout.
NDArray<DTypeTag> loadBytes(Uint8List bytes) {
  final totalBytes = bytes.lengthInBytes;
  final (:prefixLen, :headerLen) = _parseNpyPrefix(
    math.min(12, totalBytes),
    (i) => bytes[i],
  );
  final dataOffset = prefixLen + headerLen;
  if (totalBytes < dataOffset) {
    throw FormatException(
      'Unexpected EOF while reading NPY header: expected $dataOffset bytes, got $totalBytes',
    );
  }

  final headerView = Uint8List.sublistView(bytes, prefixLen, dataOffset);
  final headerStr = utf8.decode(headerView, allowMalformed: true);
  final (:dtype, :shape, :strides, :byteSize) = _prepareNpyLayout(headerStr);
  final remainingBytes = totalBytes - dataOffset;
  if (remainingBytes < byteSize) {
    throw FormatException(
      'Unexpected EOF while reading NPY payload: expected $byteSize bytes, got $remainingBytes',
    );
  }

  final result = NDArray.create(shape, dtype, strides: strides);
  if (byteSize > 0) {
    copyBytesToNative(
      result.pointer.cast<ffi.Void>(),
      Uint8List.sublistView(bytes, dataOffset, dataOffset + byteSize),
      byteSize,
    );
  }
  return result;
}

({
  ffi.Pointer<ffi.Pointer<ffi.Char>> cNames,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> cHeaderBytes,
  ffi.Pointer<ffi.Size> cHeaderLens,
  ffi.Pointer<ffi.Pointer<ffi.Void>> cDataPtrs,
  ffi.Pointer<ffi.Size> cDataLens,
})
_stageNpzInputs(
  Map<String, NDArray<DTypeTag>> arrays,
  List<NDArray<DTypeTag>> toDispose,
) {
  final numArrays = arrays.length;
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
    _writePointerAt(cNames, idx, _toNativeUtf8InScratchArena(entryName));

    final headerBytes = _encodeNpyHeader(
      effectiveArray.dtype,
      effectiveArray.shape,
    );
    final hBuf = ScratchArena.allocate<ffi.Uint8>(headerBytes.length);
    copyBytesToNative(hBuf.cast<ffi.Void>(), headerBytes, headerBytes.length);
    _writePointerAt(cHeaderBytes, idx, hBuf);
    _writeSizeAt(cHeaderLens, idx, headerBytes.length);

    final elementCount = effectiveArray.shape.isEmpty
        ? 1
        : effectiveArray.shape.reduce((x, y) => x * y);
    final byteSize = elementCount * effectiveArray.dtype.byteWidth;

    _writePointerAt(cDataPtrs, idx, effectiveArray.pointer.cast<ffi.Void>());
    _writeSizeAt(cDataLens, idx, byteSize);
    idx++;
  }

  return (
    cNames: cNames,
    cHeaderBytes: cHeaderBytes,
    cHeaderLens: cHeaderLens,
    cDataPtrs: cDataPtrs,
    cDataLens: cDataLens,
  );
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

  final toDispose = <NDArray<DTypeTag>>[];
  final marker = ScratchArena.marker;
  try {
    final staged = _stageNpzInputs(arrays, toDispose);
    final cFilepath = _toNativeUtf8InScratchArena(filepath);
    final compressLevel = compressed ? 6 : 0;

    final status = npz_save(
      cFilepath,
      arrays.length,
      staged.cNames,
      staged.cHeaderBytes,
      staged.cHeaderLens,
      staged.cDataPtrs,
      staged.cDataLens,
      compressLevel,
    );

    if (status != 0) {
      // The native encoder flags allocation failures through the thread-local
      // OOM flag; consume it here so it cannot leak into a later operation.
      checkNativeOom();
    }
    if (status == -2) {
      throw _nativeFileException('Cannot open .npz file for writing', filepath);
    }
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

/// Serializes multiple named [NDArray] instances into an in-memory NumPy `.npz`
/// ZIP archive buffer.
///
/// Produces a standard ZIP archive containing `<key>.npy` entries (stored when
/// [compressed] is `false`, or Deflate-compressed when [compressed] is `true`),
/// without touching the filesystem — suitable for Web/Wasm runtimes, network
/// transport, or custom storage backends.
///
/// **Preconditions:**
/// - [arrays] must not be empty, and no [NDArray] in [arrays] may be disposed.
///
/// **Errors & Exceptions:**
/// - Throws a [FormatException] if native ZIP archive encoding fails.
///
/// **Performance considerations:**
/// - $O(N)$ where $N$ is the total number of elements across all packed arrays.
/// - Encodes directly in native `miniz` heap memory and copies the final
///   archive once into the returned [Uint8List].
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NumPy savez reference](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
/// for details on NumPy archive formats.
Uint8List savezBytes(
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

  final toDispose = <NDArray<DTypeTag>>[];
  final marker = ScratchArena.marker;
  try {
    final staged = _stageNpzInputs(arrays, toDispose);
    final pOutBuf = ScratchArena.allocate<ffi.Pointer<ffi.Void>>(
      ffi.sizeOf<ffi.Pointer<ffi.Void>>(),
    );
    final pOutSize = ScratchArena.allocate<ffi.Size>(ffi.sizeOf<ffi.Size>());
    final compressLevel = compressed ? 6 : 0;

    final status = npz_save_mem(
      arrays.length,
      staged.cNames,
      staged.cHeaderBytes,
      staged.cHeaderLens,
      staged.cDataPtrs,
      staged.cDataLens,
      compressLevel,
      pOutBuf,
      pOutSize,
    );

    if (status != 0) {
      checkNativeOom();
      throw FormatException(
        'Failed to encode in-memory .npz zip archive (error code: $status)',
      );
    }

    final outPtr = _readPointer(pOutBuf);
    final outSize = _readSize(pOutSize);
    try {
      final result = Uint8List(outSize);
      if (outSize > 0) {
        copyNativeToBytes(result, outPtr, outSize);
      }
      return result;
    } finally {
      npz_free_buffer(outPtr);
    }
  } finally {
    ScratchArena.reset(marker);
    for (final a in toDispose) {
      a.dispose();
    }
  }
}

Map<String, NDArray<DTypeTag>> _readNpzEntries(
  ffi.Pointer<ffi.Void> handle,
  int numEntries,
) {
  final results = <String, NDArray<DTypeTag>>{};

  const nameBufLen = 4096;
  final nameBuf = ScratchArena.allocate<ffi.Char>(
    nameBufLen * ffi.sizeOf<ffi.Char>(),
  );
  const headerBufLen = 65536;
  final headerBuf = ScratchArena.allocate<ffi.Uint8>(headerBufLen);
  final pHeaderLen = ScratchArena.allocate<ffi.Size>(ffi.sizeOf<ffi.Size>());
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
          final requiredHeaderLen = _readSize(pHeaderLen);
          if (requiredHeaderLen < 0 || requiredHeaderLen > _maxNpyHeaderBytes) {
            throw FormatException(
              'NPY header length ($requiredHeaderLen) exceeds maximum allowed size ($_maxNpyHeaderBytes bytes).',
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
          throw FormatException('Invalid .npy magic header in archive entry');
        } else {
          throw FormatException(
            'Corrupted or invalid .npy entry in .npz archive (status: $infoStatus)',
          );
        }
      }

      final filename = _fromNativeUtf8(nameBuf);
      if (!filename.endsWith('.npy')) {
        continue;
      }
      final key = filename.substring(0, filename.length - 4);

      final realHeaderLen = _readSize(pHeaderLen);
      final realDataLen = _readSize(pDataLen);
      final majorVersion = activeHeaderBuf[6];
      final minorVersion = activeHeaderBuf[7];
      if (majorVersion != 1 && majorVersion != 2 && majorVersion != 3) {
        throw FormatException(
          'Unsupported .npy format version: $majorVersion.$minorVersion',
        );
      }
      final prefixLen = majorVersion >= 2 ? 12 : 10;
      final asciiLen = math.max(0, realHeaderLen - prefixLen);
      final headerBytes = Uint8List(asciiLen);
      if (asciiLen > 0) {
        copyNativeToBytes(
          headerBytes,
          (activeHeaderBuf + prefixLen).cast<ffi.Void>(),
          asciiLen,
        );
      }
      final headerStr = utf8.decode(headerBytes, allowMalformed: true);

      final (:dtype, :shape, :strides, byteSize: expectedBytes) =
          _prepareNpyLayout(headerStr);
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
        checkNativeOom();
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
  final marker = ScratchArena.marker;
  try {
    final cFilepath = _toNativeUtf8InScratchArena(filepath);
    // Probe open so that "cannot open" is reported as a FileSystemException
    // with the OS error, distinct from a FormatException for a corrupt archive.
    final probe = native_file_open_read(cFilepath);
    if (probe.address == 0) {
      throw _nativeFileException('Cannot open .npz file for reading', filepath);
    }
    native_file_close(probe);
    final pNumEntries = ScratchArena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>(),
    );

    final handle = npz_open_reader(cFilepath, pNumEntries);
    if (handle.address == 0) {
      checkNativeOom();
      throw FormatException('Invalid or corrupted .npz ZIP archive: $filepath');
    }

    try {
      return _readNpzEntries(handle, pNumEntries.value);
    } finally {
      npz_close_reader(handle);
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Deserializes multiple named [NDArray] instances from an in-memory NumPy
/// `.npz` ZIP archive buffer.
///
/// Unpacks all `.npy` entries (ignoring directories and non-`.npy` files),
/// supporting both uncompressed and Deflate-compressed archives, ZIP64, and
/// Column-Major Fortran stride mapping without touching the filesystem.
///
/// **Errors & Exceptions:**
/// - Throws a [FormatException] if [bytes] is not a valid ZIP archive or
///   contains a corrupted `.npy` entry.
/// - Throws an [UnsupportedError] if an entry specifies Big-Endian byte order
///   or an unsupported NumPy dtype descriptor.
///
/// **Performance considerations:**
/// - $O(N)$ where $N$ is the total number of bytes in the archive and unpacked
///   arrays.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates new arrays on the unmanaged C heap. The caller takes ownership
///   and must call [NDArray.dispose] on each returned array unless running
///   inside an [NDArray.scope].
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
///
/// Refer to the [NumPy load reference](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
/// for details on `.npz` archives.
Map<String, NDArray<DTypeTag>> loadzBytes(Uint8List bytes) {
  final byteLength = bytes.lengthInBytes;
  if (byteLength < 22) {
    throw FormatException(
      'Invalid or corrupted .npz ZIP archive: input too short ($byteLength bytes).',
    );
  }
  if (ffi.sizeOf<ffi.Size>() == 4 && byteLength > maxWasm32AllocationBytes) {
    throw OutOfMemoryError();
  }

  final ffi.Pointer<ffi.Uint8> nativeBuf;
  try {
    nativeBuf = malloc<ffi.Uint8>(byteLength);
    // package:ffi's malloc throws ArgumentError when OS allocation fails.
    // ignore: avoid_catching_errors
  } on ArgumentError {
    throw OutOfMemoryError();
  }
  try {
    copyBytesToNative(nativeBuf.cast<ffi.Void>(), bytes, byteLength);
    final marker = ScratchArena.marker;
    try {
      final pNumEntries = ScratchArena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>(),
      );
      final handle = npz_open_reader_mem(nativeBuf, byteLength, pNumEntries);
      if (handle.address == 0) {
        checkNativeOom();
        throw FormatException(
          'Invalid or corrupted in-memory .npz ZIP archive',
        );
      }
      try {
        return _readNpzEntries(handle, pNumEntries.value);
      } finally {
        npz_close_reader(handle);
      }
    } finally {
      ScratchArena.reset(marker);
    }
  } finally {
    malloc.free(nativeBuf);
  }
}
