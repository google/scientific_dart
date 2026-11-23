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

/// Maximum allowed JSON header size in bytes (100 MiB), matching the
/// reference Hugging Face `safetensors` implementation.
const int _maxHeaderBytes = 100 * 1024 * 1024;

/// Metadata for a single tensor stored inside a `.safetensors` archive.
final class SafetensorsTensorInfo {
  /// Tensor name (key in the `.safetensors` JSON header).
  final String name;

  /// Original `.safetensors` dtype tag (for example `'F32'`, `'F16'`, `'BF16'`).
  final String safetensorsDType;

  /// Corresponding `package:ndarray` [DType].
  final DType<DTypeTag> dtype;

  /// Tensor dimensions (unmodifiable).
  final List<int> shape;

  /// Byte offset of the tensor payload relative to the start of the data buffer.
  final int dataOffsetStart;

  /// Exclusive byte end offset relative to the start of the data buffer.
  final int dataOffsetEnd;

  /// Creates tensor metadata with an unmodifiable copy of [shape].
  SafetensorsTensorInfo({
    required this.name,
    required this.safetensorsDType,
    required this.dtype,
    required List<int> shape,
    required this.dataOffsetStart,
    required this.dataOffsetEnd,
  }) : shape = List<int>.unmodifiable(shape);

  /// Payload size in bytes.
  int get byteLength => dataOffsetEnd - dataOffsetStart;

  /// Total number of scalar elements implied by [shape].
  int get elementCount {
    var total = 1;
    for (final dim in shape) {
      total *= dim;
    }
    return total;
  }

  @override
  String toString() =>
      'SafetensorsTensorInfo($name, $safetensorsDType, shape: $shape, bytes: $byteLength)';
}

/// Reader and encoder for the Hugging Face `.safetensors` binary tensor format.
///
/// Supports both in-memory buffers ([SafetensorsFile.fromBytes]) and lazy
/// on-disk reads ([SafetensorsFile.open]) that only read requested tensor byte
/// ranges into native `NDArray` memory via [NDArray.fromBytes].
///
/// Format reference: https://huggingface.co/docs/safetensors/index
final class SafetensorsFile {
  final Map<String, SafetensorsTensorInfo> _tensors;
  final Map<String, String> _metadata;
  final int _dataBaseOffset;
  final Uint8List? _memoryBytes;
  final RandomAccessFile? _file;
  bool _isClosed = false;

  SafetensorsFile._({
    required Map<String, SafetensorsTensorInfo> tensors,
    required Map<String, String> metadata,
    required int dataBaseOffset,
    Uint8List? memoryBytes,
    RandomAccessFile? file,
  }) : _tensors = Map<String, SafetensorsTensorInfo>.unmodifiable(tensors),
       _metadata = Map<String, String>.unmodifiable(metadata),
       _dataBaseOffset = dataBaseOffset,
       _memoryBytes = memoryBytes,
       _file = file;

  /// Parses a `.safetensors` archive from an in-memory [bytes] buffer.
  ///
  /// Throws a [FormatException] if the binary framing, JSON header, dtype,
  /// shape, or byte offsets are malformed or out of bounds.
  factory SafetensorsFile.fromBytes(Uint8List bytes) {
    _requireLittleEndianHost();
    if (bytes.lengthInBytes < 8) {
      throw FormatException(
        'Safetensors buffer is too short (${bytes.lengthInBytes} bytes; need at least 8).',
      );
    }
    final bd = ByteData.sublistView(bytes, 0, 8);
    final headerLen = bd.getUint64(0, Endian.little);
    final dataSectionLength = _validateHeaderLength(
      headerLen,
      bytes.lengthInBytes,
    );
    final headerBytes = Uint8List.sublistView(bytes, 8, 8 + headerLen);
    final parsed = _parseHeader(headerBytes, dataSectionLength);
    return SafetensorsFile._(
      tensors: parsed.tensors,
      metadata: parsed.metadata,
      dataBaseOffset: 8 + headerLen,
      memoryBytes: bytes,
    );
  }

  /// Opens a `.safetensors` file on disk for lazy tensor loading.
  ///
  /// Only the JSON header is read immediately. Individual tensors are read on
  /// demand via [loadTensor] or [loadFloat32]. Call [close] when done.
  ///
  /// Throws a [FormatException] if the file header or tensor layout is invalid.
  factory SafetensorsFile.open(File file) {
    _requireLittleEndianHost();
    final raf = file.openSync(mode: FileMode.read);
    try {
      final totalLength = raf.lengthSync();
      if (totalLength < 8) {
        throw FormatException(
          'Safetensors file "${file.path}" is too short ($totalLength bytes; need at least 8).',
        );
      }
      final lenBuf = raf.readSync(8);
      if (lenBuf.length != 8) {
        throw FormatException(
          'Unexpected EOF reading header length from "${file.path}".',
        );
      }
      final headerLen = ByteData.sublistView(
        lenBuf,
      ).getUint64(0, Endian.little);
      final dataSectionLength = _validateHeaderLength(headerLen, totalLength);
      final headerBytes = raf.readSync(headerLen);
      if (headerBytes.length != headerLen) {
        throw FormatException(
          'Unexpected EOF reading $headerLen-byte header from "${file.path}".',
        );
      }
      final parsed = _parseHeader(headerBytes, dataSectionLength);
      return SafetensorsFile._(
        tensors: parsed.tensors,
        metadata: parsed.metadata,
        dataBaseOffset: 8 + headerLen,
        file: raf,
      );
    } catch (_) {
      raf.closeSync();
      rethrow;
    }
  }

  /// Optional string metadata stored under `"__metadata__"` in the header.
  Map<String, String> get metadata => _metadata;

  /// All tensor names present in the archive.
  Iterable<String> get tensorNames => _tensors.keys;

  /// All parsed tensor descriptors indexed by tensor name.
  Map<String, SafetensorsTensorInfo> get tensors => _tensors;

  /// Whether this archive has been closed via [close].
  bool get isClosed => _isClosed;

  /// Returns `true` if a tensor named [name] exists in this archive.
  bool contains(String name) => _tensors.containsKey(name);

  /// Returns metadata for the tensor named [name].
  ///
  /// It is an error if [name] is not present in this archive.
  SafetensorsTensorInfo info(String name) {
    final entry = _tensors[name];
    if (entry == null) {
      throw ArgumentError.value(
        name,
        'name',
        'Must be a tensor name present in the safetensors archive',
      );
    }
    return entry;
  }

  /// Loads the tensor named [name] with its native [DType].
  ///
  /// The returned [NDArray] is registered in the active [NDArray.scope] if one
  /// exists; otherwise the caller owns it and must call [NDArray.dispose].
  NDArray<DTypeTag> loadTensor(String name) {
    if (_isClosed) {
      throw StateError('Cannot load tensor from a closed SafetensorsFile.');
    }
    final entry = info(name);
    final payload = _readSlice(entry.dataOffsetStart, entry.byteLength);
    return _arrayFromBytes(payload, entry.shape, entry.dtype);
  }

  /// Loads the tensor named [name] as `NDArray<Float32>`.
  ///
  /// If the stored tensor is `F32`, it is bulk-copied directly into native
  /// memory via [NDArray.fromBytes]. If stored as `F16`, `BF16`, or `F64`, it
  /// is loaded in a temporary scope and cast to `DType.float32`.
  ///
  /// Throws a [FormatException] if the tensor is not a floating-point dtype.
  NDArray<Float32> loadFloat32(String name) {
    if (_isClosed) {
      throw StateError('Cannot load tensor from a closed SafetensorsFile.');
    }
    final entry = info(name);
    if (entry.dtype == DType.float32) {
      final payload = _readSlice(entry.dataOffsetStart, entry.byteLength);
      return NDArray<Float32>.fromBytes(payload, entry.shape, DType.float32);
    }
    if (entry.dtype == DType.float16 ||
        entry.dtype == DType.bfloat16 ||
        entry.dtype == DType.float64) {
      return NDArray.scope(() {
        final raw = loadTensor(name);
        return raw.asType(DType.float32).detachToParentScope();
      });
    }
    throw FormatException(
      'Tensor "$name" has dtype ${entry.safetensorsDType}, expected floating-point (F32/F16/BF16/F64).',
    );
  }

  /// Closes the underlying file handle (if opened via [SafetensorsFile.open]).
  void close() {
    if (_isClosed) return;
    _isClosed = true;
    _file?.closeSync();
  }

  Uint8List _readSlice(int relativeStart, int byteLength) {
    if (byteLength == 0) return Uint8List(0);
    final absStart = _dataBaseOffset + relativeStart;
    final mem = _memoryBytes;
    if (mem != null) {
      return Uint8List.sublistView(mem, absStart, absStart + byteLength);
    }
    final raf = _file!;
    raf.setPositionSync(absStart);
    final buf = raf.readSync(byteLength);
    if (buf.length != byteLength) {
      throw FormatException(
        'Unexpected EOF reading $byteLength bytes at offset $absStart.',
      );
    }
    return buf;
  }

  /// Encodes [tensors] and optional string [metadata] into the canonical
  /// `.safetensors` binary format.
  ///
  /// Tensor keys are sorted lexicographically so output bytes are deterministic.
  /// It is an error if any tensor uses a complex dtype, which `.safetensors`
  /// does not define, or if any tensor has been disposed.
  static Uint8List encode(
    Map<String, NDArray<DTypeTag>> tensors, {
    Map<String, String>? metadata,
  }) {
    _requireLittleEndianHost();
    final names = tensors.keys.toList()..sort();
    final headerMap = <String, Object?>{};
    if (metadata != null && metadata.isNotEmpty) {
      headerMap['__metadata__'] = Map<String, String>.of(metadata);
    }

    final chunks = <Uint8List>[];
    var cursor = 0;
    for (final name in names) {
      if (name == '__metadata__') {
        throw ArgumentError.value(
          name,
          'tensors',
          'Must not use reserved safetensors key "__metadata__"',
        );
      }
      final arr = tensors[name]!;
      if (arr.isDisposed) {
        throw StateError('Cannot encode disposed NDArray "$name".');
      }
      final tag = _dtypeToSafetensorsTag(arr.dtype);
      final bytes = arr.toBytes();
      final end = cursor + bytes.lengthInBytes;
      headerMap[name] = <String, Object?>{
        'dtype': tag,
        'shape': arr.shape,
        'data_offsets': <int>[cursor, end],
      };
      chunks.add(bytes);
      cursor = end;
    }

    final rawHeader = Uint8List.fromList(utf8.encode(jsonEncode(headerMap)));
    // Pad header with ASCII spaces to an 8-byte boundary (matching Hugging Face).
    final padLen = (8 - (rawHeader.lengthInBytes % 8)) % 8;
    final headerLen = rawHeader.lengthInBytes + padLen;
    final totalLen = 8 + headerLen + cursor;

    final out = Uint8List(totalLen);
    ByteData.sublistView(out, 0, 8).setUint64(0, headerLen, Endian.little);
    out.setRange(8, 8 + rawHeader.lengthInBytes, rawHeader);
    for (var i = 0; i < padLen; i++) {
      out[8 + rawHeader.lengthInBytes + i] = 0x20;
    }

    var writeOffset = 8 + headerLen;
    for (final chunk in chunks) {
      out.setRange(writeOffset, writeOffset + chunk.lengthInBytes, chunk);
      writeOffset += chunk.lengthInBytes;
    }
    return out;
  }

  /// Writes [tensors] and optional [metadata] to [file] in `.safetensors` format.
  static void writeToFile(
    File file,
    Map<String, NDArray<DTypeTag>> tensors, {
    Map<String, String>? metadata,
  }) {
    final encoded = encode(tensors, metadata: metadata);
    file.parent.createSync(recursive: true);
    file.writeAsBytesSync(encoded, flush: true);
  }
}

void _requireLittleEndianHost() {
  if (Endian.host != Endian.little) {
    throw UnsupportedError(
      'SafetensorsFile requires a little-endian host platform.',
    );
  }
}

int _validateHeaderLength(int headerLen, int totalLength) {
  if (headerLen <= 0 || headerLen > _maxHeaderBytes) {
    throw FormatException(
      'Invalid safetensors header length: $headerLen bytes (allowed 1..$_maxHeaderBytes).',
    );
  }
  if (headerLen > totalLength - 8) {
    throw FormatException(
      'Safetensors header length ($headerLen) exceeds remaining file length (${totalLength - 8}).',
    );
  }
  return totalLength - 8 - headerLen;
}

({Map<String, SafetensorsTensorInfo> tensors, Map<String, String> metadata})
_parseHeader(Uint8List headerBytes, int dataSectionLength) {
  final String headerText;
  try {
    headerText = utf8.decode(headerBytes, allowMalformed: false);
  } on FormatException catch (e) {
    throw FormatException('Invalid UTF-8 in safetensors header: ${e.message}');
  }

  final Object? decoded;
  try {
    decoded = jsonDecode(headerText);
  } on FormatException catch (e) {
    throw FormatException('Invalid JSON in safetensors header: ${e.message}');
  }

  if (decoded is! Map<String, dynamic>) {
    throw const FormatException('Safetensors header JSON must be an object.');
  }

  final metadata = <String, String>{};
  final tensors = <String, SafetensorsTensorInfo>{};

  for (final entry in decoded.entries) {
    final key = entry.key;
    final value = entry.value;
    if (key == '__metadata__') {
      if (value == null) continue;
      if (value is! Map<String, dynamic>) {
        throw const FormatException(
          'Safetensors "__metadata__" entry must be a JSON object of strings.',
        );
      }
      for (final m in value.entries) {
        if (m.value is! String) {
          throw FormatException(
            'Safetensors metadata value for "${m.key}" must be a string.',
          );
        }
        metadata[m.key] = m.value as String;
      }
      continue;
    }

    if (value is! Map<String, dynamic>) {
      throw FormatException(
        'Safetensors tensor entry "$key" must be a JSON object.',
      );
    }
    final dtypeRaw = value['dtype'];
    final shapeRaw = value['shape'];
    final offsetsRaw = value['data_offsets'];
    if (dtypeRaw is! String) {
      throw FormatException('Tensor "$key" missing string "dtype".');
    }
    if (shapeRaw is! List<dynamic>) {
      throw FormatException('Tensor "$key" missing list "shape".');
    }
    if (offsetsRaw is! List<dynamic> || offsetsRaw.length != 2) {
      throw FormatException(
        'Tensor "$key" must have a 2-element "data_offsets" list.',
      );
    }

    final dtype = _safetensorsTagToDType(dtypeRaw, key);
    final shape = <int>[];
    var elementCount = 1;
    const maxInt = 0x7fffffffffffffff;
    for (final d in shapeRaw) {
      if (d is! int || d < 0) {
        throw FormatException(
          'Tensor "$key" contains invalid dimension "$d" in shape.',
        );
      }
      shape.add(d);
      if (d == 0) {
        elementCount = 0;
      } else if (elementCount > maxInt ~/ d) {
        throw FormatException('Tensor "$key" shape $shape overflows 64 bits.');
      } else {
        elementCount *= d;
      }
    }

    final startRaw = offsetsRaw[0];
    final endRaw = offsetsRaw[1];
    if (startRaw is! int || endRaw is! int) {
      throw FormatException(
        'Tensor "$key" data_offsets must be integers: $offsetsRaw',
      );
    }
    if (startRaw < 0 || endRaw < startRaw || endRaw > dataSectionLength) {
      throw FormatException(
        'Tensor "$key" data_offsets [$startRaw, $endRaw] out of bounds for $dataSectionLength-byte data section.',
      );
    }

    if (elementCount > maxInt ~/ dtype.byteWidth) {
      throw FormatException(
        'Tensor "$key" byte length overflows 64-bit integer.',
      );
    }
    final expectedBytes = elementCount * dtype.byteWidth;
    final actualBytes = endRaw - startRaw;
    if (actualBytes != expectedBytes) {
      throw FormatException(
        'Tensor "$key" byte length mismatch: offsets span $actualBytes bytes, '
        'but shape $shape with $dtypeRaw requires $expectedBytes bytes.',
      );
    }

    tensors[key] = SafetensorsTensorInfo(
      name: key,
      safetensorsDType: dtypeRaw,
      dtype: dtype,
      shape: shape,
      dataOffsetStart: startRaw,
      dataOffsetEnd: endRaw,
    );
  }

  // Verify non-overlapping byte intervals (empty tensors are allowed anywhere).
  final nonEmpty = tensors.values.where((t) => t.byteLength > 0).toList()
    ..sort((a, b) => a.dataOffsetStart.compareTo(b.dataOffsetStart));
  for (var i = 1; i < nonEmpty.length; i++) {
    final prev = nonEmpty[i - 1];
    final curr = nonEmpty[i];
    if (curr.dataOffsetStart < prev.dataOffsetEnd) {
      throw FormatException(
        'Overlapping safetensors data offsets between "${prev.name}" '
        '[${prev.dataOffsetStart}, ${prev.dataOffsetEnd}) and "${curr.name}" '
        '[${curr.dataOffsetStart}, ${curr.dataOffsetEnd}).',
      );
    }
  }

  return (tensors: tensors, metadata: metadata);
}

DType<DTypeTag> _safetensorsTagToDType(String tag, String tensorName) =>
    switch (tag) {
      'F64' => DType.float64,
      'F32' => DType.float32,
      'F16' => DType.float16,
      'BF16' => DType.bfloat16,
      'I64' => DType.int64,
      'I32' => DType.int32,
      'I16' => DType.int16,
      'I8' => DType.int8,
      'U64' => DType.uint64,
      'U32' => DType.uint32,
      'U16' => DType.uint16,
      'U8' => DType.uint8,
      'BOOL' => DType.boolean,
      _ => throw FormatException(
        'Unsupported safetensors dtype "$tag" for tensor "$tensorName".',
      ),
    };

String _dtypeToSafetensorsTag(DType<DTypeTag> dtype) => switch (dtype) {
  DType.float64 => 'F64',
  DType.float32 => 'F32',
  DType.float16 => 'F16',
  DType.bfloat16 => 'BF16',
  DType.int64 => 'I64',
  DType.int32 => 'I32',
  DType.int16 => 'I16',
  DType.int8 => 'I8',
  DType.uint64 => 'U64',
  DType.uint32 => 'U32',
  DType.uint16 => 'U16',
  DType.uint8 => 'U8',
  DType.boolean => 'BOOL',
  DType.complex128 || DType.complex64 => throw UnsupportedError(
    'Safetensors format does not support complex dtype $dtype.',
  ),
};

NDArray<DTypeTag> _arrayFromBytes(
  Uint8List bytes,
  List<int> shape,
  DType<DTypeTag> dtype,
) => switch (dtype) {
  DType.float64 => NDArray<Float64>.fromBytes(bytes, shape, DType.float64),
  DType.float32 => NDArray<Float32>.fromBytes(bytes, shape, DType.float32),
  DType.float16 => NDArray<Float16>.fromBytes(bytes, shape, DType.float16),
  DType.bfloat16 => NDArray<BFloat16>.fromBytes(bytes, shape, DType.bfloat16),
  DType.int64 => NDArray<Int64>.fromBytes(bytes, shape, DType.int64),
  DType.int32 => NDArray<Int32>.fromBytes(bytes, shape, DType.int32),
  DType.int16 => NDArray<Int16>.fromBytes(bytes, shape, DType.int16),
  DType.int8 => NDArray<Int8>.fromBytes(bytes, shape, DType.int8),
  DType.uint64 => NDArray<Uint64>.fromBytes(bytes, shape, DType.uint64),
  DType.uint32 => NDArray<Uint32>.fromBytes(bytes, shape, DType.uint32),
  DType.uint16 => NDArray<Uint16>.fromBytes(bytes, shape, DType.uint16),
  DType.uint8 => NDArray<Uint8>.fromBytes(bytes, shape, DType.uint8),
  DType.boolean => NDArray<Boolean>.fromBytes(bytes, shape, DType.boolean),
  DType.complex128 || DType.complex64 => throw UnsupportedError(
    'Safetensors format does not support complex dtype $dtype.',
  ),
};
