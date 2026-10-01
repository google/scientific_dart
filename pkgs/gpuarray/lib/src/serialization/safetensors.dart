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

GpuArray<DTypeTag> _createEmptyTypedTensor(
  List<int> shape,
  DType dtype,
  GpuDevice device,
) => switch (dtype) {
  DType.float64 => GpuArray<Float64>.empty(
    shape,
    DType.float64,
    device: device,
  ),
  DType.float32 => GpuArray<Float32>.empty(
    shape,
    DType.float32,
    device: device,
  ),
  DType.float16 => GpuArray<Float16>.empty(
    shape,
    DType.float16,
    device: device,
  ),
  DType.bfloat16 => GpuArray<BFloat16>.empty(
    shape,
    DType.bfloat16,
    device: device,
  ),
  DType.int64 => GpuArray<Int64>.empty(shape, DType.int64, device: device),
  DType.int32 => GpuArray<Int32>.empty(shape, DType.int32, device: device),
  DType.int16 => GpuArray<Int16>.empty(shape, DType.int16, device: device),
  DType.int8 => GpuArray<Int8>.empty(shape, DType.int8, device: device),
  DType.uint64 => GpuArray<Uint64>.empty(shape, DType.uint64, device: device),
  DType.uint32 => GpuArray<Uint32>.empty(shape, DType.uint32, device: device),
  DType.uint16 => GpuArray<Uint16>.empty(shape, DType.uint16, device: device),
  DType.uint8 => GpuArray<Uint8>.empty(shape, DType.uint8, device: device),
  DType.boolean => GpuArray<Bool>.empty(shape, DType.boolean, device: device),
  DType.complex64 => GpuArray<Complex64>.empty(
    shape,
    DType.complex64,
    device: device,
  ),
  DType.complex128 => GpuArray<Complex128>.empty(
    shape,
    DType.complex128,
    device: device,
  ),
};

/// Serializes a map of named [tensors] and optional string [metadata] into the
/// binary SafeTensors format.
///
/// Synchronizes any pending GPU writes on each tensor's buffer to host memory
/// before reading its raw byte payload. Non-contiguous tensor views are
/// materialized in an isolated scope that is disposed before returning.
///
/// None of the arrays in [tensors] may be disposed, and no key in [tensors]
/// may be the reserved `'__metadata__'` key.
Uint8List saveSafetensors(
  Map<String, GpuArray<DTypeTag>> tensors, {
  Map<String, String>? metadata,
}) {
  final snapshot = Map<String, GpuArray<DTypeTag>>.of(tensors);
  for (final entry in snapshot.entries) {
    if (entry.key == '__metadata__') {
      throw ArgumentError.value(
        entry.key,
        'tensors',
        'Must not use reserved key "__metadata__" as a tensor name.',
      );
    }
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

    final contiguousTensor = tensor.isContiguous ? tensor : tensor.copy();
    try {
      final byteLength =
          contiguousTensor.size * contiguousTensor.dtype.byteWidth;
      headerMap[name] = <String, Object>{
        'dtype': _dtypeToSafetensors(contiguousTensor.dtype),
        'shape': List<int>.of(contiguousTensor.shape),
        'data_offsets': <int>[currentOffset, currentOffset + byteLength],
      };
      currentOffset += byteLength;

      final byteOffset =
          contiguousTensor.offsetElements * contiguousTensor.dtype.byteWidth;
      final tensorBytes = byteLength > 0
          ? contiguousTensor.buffer.readBytes(
              offset: byteOffset,
              bytes: byteLength,
            )
          : Uint8List(0);
      tensorPayloads.add(tensorBytes);
    } finally {
      if (!identical(contiguousTensor, tensor)) {
        contiguousTensor.dispose();
      }
    }
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
/// Uploads each loaded tensor's bytes to its underlying [GpuBuffer] via
/// `copyFromHost` so subsequent GPU shader dispatches see the deserialized
/// weights.
///
/// The target [device] must not be disposed.
/// Throws a [FormatException] if [bytes] is truncated, has a malformed UTF-8
/// or JSON header, specifies an unknown dtype, contains negative dimensions,
/// or has invalid or out-of-bounds `data_offsets`.
Map<String, GpuArray<DTypeTag>> loadSafetensors(
  Uint8List bytes, {
  GpuDevice? device,
}) {
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
    throw FormatException('Invalid SafeTensors UTF-8 header: ${error.message}');
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
  final loadedTensors = <String, GpuArray<DTypeTag>>{};

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

      final tensor = _createEmptyTypedTensor(shape, dtype, targetDevice);
      if (expectedByteSize > 0) {
        final sourceView = Uint8List.sublistView(
          bytes,
          absoluteStart,
          absoluteEnd,
        );
        tensor.buffer.writeBytes(sourceView);
      }

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
  Map<String, GpuArray<DTypeTag>> tensors, {
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
Map<String, GpuArray<DTypeTag>> loadSafetensorsFile(
  String filePath, {
  GpuDevice? device,
}) {
  final bytes = File(filePath).readAsBytesSync();
  return loadSafetensors(bytes, device: device);
}
