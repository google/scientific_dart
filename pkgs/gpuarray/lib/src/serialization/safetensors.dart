import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:resource_scope/resource_scope.dart';

import '../device.dart';
import '../dtype.dart';
import '../exceptions.dart';
import '../gpu_array.dart';

/// Maps a [DType] to its SafeTensors specification identifier string.
String _dtypeToSafetensors(DType dtype) => switch (dtype) {
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
  DType.complex64 => 'C64',
  DType.complex128 => 'C128',
};

/// Maps a SafeTensors dtype string [code] to its corresponding [DType].
///
/// Throws a [FormatException] if [code] is not a recognized SafeTensors dtype.
DType _safetensorsToDtype(String code) => switch (code) {
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
  'C64' => DType.complex64,
  'C128' => DType.complex128,
  _ => throw FormatException('Unsupported SafeTensors dtype: "$code".'),
};

/// Serializes a map of named [tensors] and optional string [metadata] into the
/// binary SafeTensors format.
///
/// Synchronizes any pending GPU writes on each tensor's buffer to host memory
/// before reading its raw byte payload. Non-contiguous tensor views are
/// materialized in an isolated [ResourceScope] that is disposed before
/// returning.
///
/// None of the arrays in [tensors] may be disposed.
Uint8List saveSafetensors(
  Map<String, GpuArray> tensors, {
  Map<String, String>? metadata,
}) {
  final snapshot = Map<String, GpuArray>.of(tensors);
  for (final entry in snapshot.entries) {
    if (entry.value.isDisposed) {
      throw StateError(
        'Cannot serialize disposed GpuArray "${entry.key}" to SafeTensors.',
      );
    }
  }

  final headerMap = <String, Object>{};
  if (metadata != null) {
    headerMap['__metadata__'] = Map<String, String>.of(metadata);
  }

  var currentOffset = 0;
  final tensorPayloads = <Uint8List>[];

  for (final entry in snapshot.entries) {
    final name = entry.key;
    final tensor = entry.value;

    final payload = ResourceScope.scope(() {
      final contiguousTensor = tensor.isContiguous ? tensor : tensor.copy();
      contiguousTensor.buffer.ensureHostSynced();
      final byteLength = contiguousTensor.byteSize;

      headerMap[name] = <String, Object>{
        'dtype': _dtypeToSafetensors(contiguousTensor.dtype),
        'shape': List<int>.of(contiguousTensor.shape),
        'data_offsets': <int>[currentOffset, currentOffset + byteLength],
      };

      currentOffset += byteLength;

      final tensorBytes = Uint8List(byteLength);
      if (byteLength > 0) {
        final byteOffset =
            contiguousTensor.offsetElements * contiguousTensor.dtype.byteWidth;
        final sourceAddress = contiguousTensor.buffer.address + byteOffset;
        tensorBytes.setAll(0, sourceAddress.asTypedList(byteLength));
      }
      return tensorBytes;
    });

    tensorPayloads.add(payload);
  }

  final headerJson = jsonEncode(headerMap);
  final rawHeaderBytes = utf8.encode(headerJson);
  final paddingLength = (8 - (rawHeaderBytes.length % 8)) % 8;
  final headerLength = rawHeaderBytes.length + paddingLength;

  final totalSize = 8 + headerLength + currentOffset;
  final outputBytes = Uint8List(totalSize);
  final byteData = ByteData.sublistView(outputBytes);

  byteData.setUint64(0, headerLength, Endian.little);
  outputBytes.setAll(8, rawHeaderBytes);
  for (var i = 0; i < paddingLength; i++) {
    outputBytes[8 + rawHeaderBytes.length + i] = 0x20;
  }

  var writeOffset = 8 + headerLength;
  for (final tensorBytes in tensorPayloads) {
    outputBytes.setAll(writeOffset, tensorBytes);
    writeOffset += tensorBytes.length;
  }

  return outputBytes;
}

/// Deserializes a binary SafeTensors buffer [bytes] into a map of named
/// [GpuArray] tensors allocated on [device] (or [GpuDevice.defaultDevice]).
///
/// Marks each loaded tensor's underlying [GpuBuffer] as host-modified so
/// subsequent GPU shader dispatches upload the deserialized weights.
///
/// The target [device] must not be disposed.
/// Throws a [FormatException] if [bytes] is truncated, has a malformed UTF-8
/// or JSON header, specifies an unknown dtype, contains negative dimensions,
/// or has invalid or out-of-bounds `data_offsets`.
Map<String, GpuArray> loadSafetensors(Uint8List bytes, {GpuDevice? device}) {
  final targetDevice = device ?? GpuDevice.defaultDevice;
  if (targetDevice.isDisposed) {
    throw GpuDeviceDisposedException(targetDevice.name);
  }
  if (bytes.length < 8) {
    throw FormatException(
      'Invalid SafeTensors buffer: byte length (${bytes.length}) is smaller '
      'than the 8-byte header size prefix.',
    );
  }

  final byteData = ByteData.sublistView(bytes);
  final headerLength = byteData.getUint64(0, Endian.little);
  if (headerLength < 0 ||
      headerLength > bytes.length - 8 ||
      8 + headerLength > bytes.length) {
    throw FormatException(
      'Invalid SafeTensors buffer: header length ($headerLength) exceeds '
      'available buffer size (${bytes.length - 8}).',
    );
  }

  final headerSlice = Uint8List.sublistView(bytes, 8, 8 + headerLength);
  final String headerJson;
  try {
    headerJson = utf8.decode(headerSlice);
  } on FormatException catch (error) {
    throw FormatException(
      'Invalid SafeTensors UTF-8 header: ${error.message}',
    );
  }

  final Object? decodedHeader;
  try {
    decodedHeader = jsonDecode(headerJson);
  } on FormatException catch (error) {
    throw FormatException('Invalid SafeTensors JSON header: ${error.message}');
  }

  if (decodedHeader is! Map<String, dynamic>) {
    throw const FormatException(
      'Invalid SafeTensors header: root JSON value must be an object.',
    );
  }

  final dataStartOffset = 8 + headerLength;
  final loadedTensors = <String, GpuArray>{};

  try {
    for (final entry in decodedHeader.entries) {
      final tensorName = entry.key;
      final descriptor = entry.value;
      if (tensorName == '__metadata__') {
        if (descriptor is! Map) {
          throw const FormatException(
            'Invalid SafeTensors "__metadata__" entry: must be a JSON object.',
          );
        }
        continue;
      }

      if (descriptor is! Map<String, dynamic>) {
        throw FormatException(
          'Invalid SafeTensors descriptor for tensor "$tensorName": '
          'must be a JSON object.',
        );
      }

      final rawDtype = descriptor['dtype'];
      if (rawDtype is! String) {
        throw FormatException(
          'Missing or non-string "dtype" for tensor "$tensorName".',
        );
      }
      final dtype = _safetensorsToDtype(rawDtype);

      final rawShape = descriptor['shape'];
      if (rawShape is! List) {
        throw FormatException(
          'Missing or non-list "shape" for tensor "$tensorName".',
        );
      }
      final shape = <int>[];
      var elementCount = 1;
      for (final dimension in rawShape) {
        if (dimension is! int || dimension < 0) {
          throw FormatException(
            'Invalid shape dimension "$dimension" for tensor "$tensorName": '
            'dimensions must be non-negative integers.',
          );
        }
        shape.add(dimension);
        elementCount *= dimension;
      }

      final rawOffsets = descriptor['data_offsets'];
      if (rawOffsets is! List || rawOffsets.length != 2) {
        throw FormatException(
          'Invalid "data_offsets" for tensor "$tensorName": '
          'expected a 2-element integer list.',
        );
      }
      final startOffset = rawOffsets[0];
      final endOffset = rawOffsets[1];
      if (startOffset is! int ||
          endOffset is! int ||
          startOffset < 0 ||
          endOffset < startOffset) {
        throw FormatException(
          'Invalid "data_offsets" [$startOffset, $endOffset] for tensor '
          '"$tensorName".',
        );
      }

      final expectedByteSize = elementCount * dtype.byteWidth;
      if (endOffset - startOffset != expectedByteSize) {
        throw FormatException(
          'Data offset span (${endOffset - startOffset}) does not match '
          'expected byte size ($expectedByteSize) for tensor "$tensorName".',
        );
      }

      final absoluteStart = dataStartOffset + startOffset;
      final absoluteEnd = dataStartOffset + endOffset;
      if (absoluteEnd > bytes.length) {
        throw FormatException(
          'Data offset end ($absoluteEnd) exceeds total buffer length '
          '(${bytes.length}) for tensor "$tensorName".',
        );
      }

      final tensor = GpuArray.empty(shape, dtype, device: targetDevice);
      if (expectedByteSize > 0) {
        final sourceView = Uint8List.sublistView(
          bytes,
          absoluteStart,
          absoluteEnd,
        );
        tensor.buffer.address
            .asTypedList(expectedByteSize)
            .setAll(0, sourceView);
      }
      tensor.buffer.markHostModified();

      loadedTensors[tensorName] = tensor;
    }
  } catch (_) {
    for (final allocated in loadedTensors.values) {
      allocated.dispose();
    }
    rethrow;
  }

  return loadedTensors;
}

/// Saves [tensors] and optional [metadata] to a `.safetensors` file at
/// [filePath].
///
/// None of the arrays in [tensors] may be disposed.
void saveSafetensorsFile(
  String filePath,
  Map<String, GpuArray> tensors, {
  Map<String, String>? metadata,
}) {
  final bytes = saveSafetensors(tensors, metadata: metadata);
  File(filePath).writeAsBytesSync(bytes);
}

/// Loads a map of named [GpuArray] tensors from a `.safetensors` file at
/// [filePath] onto [device] (or [GpuDevice.defaultDevice]).
///
/// Throws a [FormatException] if the file contents do not conform to the
/// SafeTensors binary specification.
Map<String, GpuArray> loadSafetensorsFile(
  String filePath, {
  GpuDevice? device,
}) {
  final bytes = File(filePath).readAsBytesSync();
  return loadSafetensors(bytes, device: device);
}

