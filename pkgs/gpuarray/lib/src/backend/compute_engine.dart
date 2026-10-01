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
import 'dart:math' as math;
import 'package:ffi/ffi.dart';
import 'package:meta/meta.dart';
import '../buffer.dart';
import '../dtype.dart';
import '../exceptions.dart';

/// Computes default C-contiguous strides (in element counts) for [shape].
List<int> computeCStrides(List<int> shape) {
  if (shape.isEmpty) return [];
  final strides = List<int>.filled(shape.length, 1);
  for (var i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/// Calculates total number of elements in a tensor of [shape].
///
/// Each dimension in [shape] must be non-negative.
int computeSize(List<int> shape) {
  if (shape.isEmpty) return 1;
  var size = 1;
  for (final dim in shape) {
    if (dim < 0) {
      throw ArgumentError.value(
        shape,
        'shape',
        'Must not contain negative dimensions.',
      );
    }
    size *= dim;
  }
  return size;
}

/// Checks whether shapes [a] and [b] have identical dimensions.
bool areShapesEqual(List<int> a, List<int> b) {
  if (identical(a, b)) return true;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// Checks whether [shape] and [strides] represent a contiguous C-order layout.
bool isContiguousLayout(List<int> shape, List<int> strides) {
  if (shape.isEmpty) return true;
  final expectedStrides = computeCStrides(shape);
  for (var i = 0; i < shape.length; i++) {
    if (shape[i] > 1 && strides[i] != expectedStrides[i]) {
      return false;
    }
  }
  return true;
}

/// Broadcasts two tensor shapes [shapeA] and [shapeB] according to NumPy broadcasting rules.
List<int> broadcastShapes(List<int> shapeA, List<int> shapeB) {
  final rankA = shapeA.length;
  final rankB = shapeB.length;
  final maxRank = math.max(rankA, rankB);
  final result = List<int>.filled(maxRank, 0);

  for (var i = 0; i < maxRank; i++) {
    final dimA = (i < maxRank - rankA) ? 1 : shapeA[i - (maxRank - rankA)];
    final dimB = (i < maxRank - rankB) ? 1 : shapeB[i - (maxRank - rankB)];

    if (dimA == dimB) {
      result[i] = dimA;
    } else if (dimA == 1) {
      result[i] = dimB;
    } else if (dimB == 1) {
      result[i] = dimA;
    } else {
      throw GpuShapeMismatchException('broadcast', shapeA, shapeB);
    }
  }
  return result;
}

/// Computes broadcasted strides for an array with [shape] and [strides] expanded to [targetShape].
List<int> broadcastStrides(
  List<int> shape,
  List<int> strides,
  List<int> targetShape,
) {
  final rank = shape.length;
  final targetRank = targetShape.length;
  if (targetRank < rank) {
    throw GpuShapeMismatchException('broadcast_to', shape, targetShape);
  }
  final result = List<int>.filled(targetRank, 0);

  final rankDiff = targetRank - rank;
  for (var i = 0; i < targetRank; i++) {
    if (i < rankDiff) {
      result[i] = 0;
    } else {
      final origDim = shape[i - rankDiff];
      final targetDim = targetShape[i];
      if (origDim == targetDim) {
        result[i] = strides[i - rankDiff];
      } else if (origDim == 1 && targetDim > 1) {
        result[i] = 0;
      } else {
        throw GpuShapeMismatchException('broadcast_to', shape, targetShape);
      }
    }
  }
  return result;
}

/// Reads a single typed value at [elementIndex] from [pointer].
Object readPointerAny(
  ffi.Pointer<ffi.Uint8> pointer,
  DType dtype,
  int elementIndex, {
  int offsetElements = 0,
}) {
  final elementOffset = offsetElements + elementIndex;
  switch (dtype) {
    case DType.float64:
      return pointer.cast<ffi.Double>()[elementOffset];
    case DType.float32:
      return pointer.cast<ffi.Float>()[elementOffset];
    case DType.float16:
      return Float16Utils.decodeFloat16(
        pointer.cast<ffi.Uint16>()[elementOffset],
      );
    case DType.bfloat16:
      return Float16Utils.decodeBFloat16(
        pointer.cast<ffi.Uint16>()[elementOffset],
      );
    case DType.int64:
      return pointer.cast<ffi.Int64>()[elementOffset];
    case DType.int32:
      return pointer.cast<ffi.Int32>()[elementOffset];
    case DType.int16:
      return pointer.cast<ffi.Int16>()[elementOffset];
    case DType.int8:
      return pointer.cast<ffi.Int8>()[elementOffset];
    case DType.uint64:
      return pointer.cast<ffi.Uint64>()[elementOffset];
    case DType.uint32:
      return pointer.cast<ffi.Uint32>()[elementOffset];
    case DType.uint16:
      return pointer.cast<ffi.Uint16>()[elementOffset];
    case DType.uint8:
      return pointer.cast<ffi.Uint8>()[elementOffset];
    case DType.boolean:
      return pointer.cast<ffi.Uint8>()[elementOffset] != 0;
    case DType.complex64:
      final real = pointer.cast<ffi.Float>()[elementOffset * 2];
      final imag = pointer.cast<ffi.Float>()[elementOffset * 2 + 1];
      return Complex(real, imag);
    case DType.complex128:
      final real = pointer.cast<ffi.Double>()[elementOffset * 2];
      final imag = pointer.cast<ffi.Double>()[elementOffset * 2 + 1];
      return Complex(real, imag);
  }
}

/// Writes a single typed [value] at [elementIndex] to [pointer].
void writePointerAny(
  ffi.Pointer<ffi.Uint8> pointer,
  DType dtype,
  int elementIndex,
  Object? value, {
  int offsetElements = 0,
}) {
  if (value is! num &&
      value is! bool &&
      value is! BigInt &&
      value is! Complex) {
    throw ArgumentError.value(
      value,
      'value',
      'Must be a numeric, boolean, BigInt, or Complex value.',
    );
  }
  final elementOffset = offsetElements + elementIndex;
  double toDoubleVal(Object? v) {
    if (v is bool) return v ? 1.0 : 0.0;
    if (v is num) return v.toDouble();
    if (v is Complex) return v.real;
    return 0.0;
  }

  int toIntVal(Object? v) {
    if (v is bool) return v ? 1 : 0;
    if (v is num) return v.toInt();
    if (v is BigInt) return v.toSigned(64).toInt();
    return 0;
  }

  switch (dtype) {
    case DType.float64:
      pointer.cast<ffi.Double>()[elementOffset] = toDoubleVal(value);
    case DType.float32:
      pointer.cast<ffi.Float>()[elementOffset] = toDoubleVal(value);
    case DType.float16:
      pointer.cast<ffi.Uint16>()[elementOffset] = Float16Utils.encodeFloat16(
        toDoubleVal(value),
      );
    case DType.bfloat16:
      pointer.cast<ffi.Uint16>()[elementOffset] = Float16Utils.encodeBFloat16(
        toDoubleVal(value),
      );
    case DType.int64:
      pointer.cast<ffi.Int64>()[elementOffset] = toIntVal(value);
    case DType.int32:
      pointer.cast<ffi.Int32>()[elementOffset] = toIntVal(value);
    case DType.int16:
      pointer.cast<ffi.Int16>()[elementOffset] = toIntVal(value);
    case DType.int8:
      pointer.cast<ffi.Int8>()[elementOffset] = toIntVal(value);
    case DType.uint64:
      pointer.cast<ffi.Uint64>()[elementOffset] = (value is BigInt)
          ? value.toUnsigned(64).toInt()
          : toIntVal(value);
    case DType.uint32:
      pointer.cast<ffi.Uint32>()[elementOffset] = toIntVal(value);
    case DType.uint16:
      pointer.cast<ffi.Uint16>()[elementOffset] = toIntVal(value);
    case DType.uint8:
      pointer.cast<ffi.Uint8>()[elementOffset] = toIntVal(value);
    case DType.boolean:
      pointer.cast<ffi.Uint8>()[elementOffset] =
          (value == true || (value is num && value != 0)) ? 1 : 0;
    case DType.complex64:
      if (value is Complex) {
        pointer.cast<ffi.Float>()[elementOffset * 2] = value.real;
        pointer.cast<ffi.Float>()[elementOffset * 2 + 1] = value.imag;
      } else {
        pointer.cast<ffi.Float>()[elementOffset * 2] = toDoubleVal(value);
        pointer.cast<ffi.Float>()[elementOffset * 2 + 1] = 0.0;
      }
    case DType.complex128:
      if (value is Complex) {
        pointer.cast<ffi.Double>()[elementOffset * 2] = value.real;
        pointer.cast<ffi.Double>()[elementOffset * 2 + 1] = value.imag;
      } else {
        pointer.cast<ffi.Double>()[elementOffset * 2] = toDoubleVal(value);
        pointer.cast<ffi.Double>()[elementOffset * 2 + 1] = 0.0;
      }
  }
}

/// Reads a single numerical value at [elementIndex] from [pointer].
double readPointerValue(
  ffi.Pointer<ffi.Uint8> pointer,
  DType dtype,
  int elementIndex, {
  int offsetElements = 0,
}) {
  final elementOffset = offsetElements + elementIndex;
  switch (dtype) {
    case DType.float64:
      return pointer.cast<ffi.Double>()[elementOffset];
    case DType.float32:
      return pointer.cast<ffi.Float>()[elementOffset];
    case DType.float16:
      return Float16Utils.decodeFloat16(
        pointer.cast<ffi.Uint16>()[elementOffset],
      );
    case DType.bfloat16:
      return Float16Utils.decodeBFloat16(
        pointer.cast<ffi.Uint16>()[elementOffset],
      );
    case DType.int64:
      return pointer.cast<ffi.Int64>()[elementOffset].toDouble();
    case DType.int32:
      return pointer.cast<ffi.Int32>()[elementOffset].toDouble();
    case DType.int16:
      return pointer.cast<ffi.Int16>()[elementOffset].toDouble();
    case DType.int8:
      return pointer.cast<ffi.Int8>()[elementOffset].toDouble();
    case DType.uint64:
      final raw = pointer.cast<ffi.Uint64>()[elementOffset];
      return raw >= 0
          ? raw.toDouble()
          : BigInt.from(raw).toUnsigned(64).toDouble();
    case DType.uint32:
      return pointer.cast<ffi.Uint32>()[elementOffset].toDouble();
    case DType.uint16:
      return pointer.cast<ffi.Uint16>()[elementOffset].toDouble();
    case DType.uint8:
      return pointer.cast<ffi.Uint8>()[elementOffset].toDouble();
    case DType.boolean:
      return pointer.cast<ffi.Uint8>()[elementOffset] != 0 ? 1.0 : 0.0;
    case DType.complex64:
      return pointer.cast<ffi.Float>()[elementOffset * 2];
    case DType.complex128:
      return pointer.cast<ffi.Double>()[elementOffset * 2];
  }
}

/// Reads a single typed value at [elementIndex] from [buffer].
Object readBufferAny(
  GpuBuffer buffer,
  DType dtype,
  int elementIndex, {
  int offsetElements = 0,
}) {
  final elementOffset = offsetElements + elementIndex;
  final byteWidth = dtype.byteWidth;
  final byteOffset = elementOffset * byteWidth;
  return using((arena) {
    final staging = arena<ffi.Uint8>(byteWidth);
    buffer.copyToHost(staging.cast<ffi.Void>(), byteWidth, offset: byteOffset);
    return readPointerAny(staging, dtype, 0);
  });
}

/// Writes a single typed [value] at [elementIndex] to [buffer].
void writeBufferAny(
  GpuBuffer buffer,
  DType dtype,
  int elementIndex,
  Object? value, {
  int offsetElements = 0,
}) {
  final elementOffset = offsetElements + elementIndex;
  final byteWidth = dtype.byteWidth;
  final byteOffset = elementOffset * byteWidth;
  using((arena) {
    final staging = arena<ffi.Uint8>(byteWidth);
    writePointerAny(staging, dtype, 0, value);
    buffer.copyFromHost(
      staging.cast<ffi.Void>(),
      byteWidth,
      offset: byteOffset,
    );
  });
}

/// Reads a single numerical value at [elementIndex] from [buffer].
double readBufferValue(
  GpuBuffer buffer,
  DType dtype,
  int elementIndex, {
  int offsetElements = 0,
}) {
  final elementOffset = offsetElements + elementIndex;
  final byteWidth = dtype.byteWidth;
  final byteOffset = elementOffset * byteWidth;
  return using((arena) {
    final staging = arena<ffi.Uint8>(byteWidth);
    buffer.copyToHost(staging.cast<ffi.Void>(), byteWidth, offset: byteOffset);
    return readPointerValue(staging, dtype, 0);
  });
}

/// Writes a single numerical [value] at [elementIndex] to [buffer].
void writeBufferValue(
  GpuBuffer buffer,
  DType dtype,
  int elementIndex,
  double value, {
  int offsetElements = 0,
}) {
  writeBufferAny(
    buffer,
    dtype,
    elementIndex,
    value,
    offsetElements: offsetElements,
  );
}

/// Internal shape and stride calculation utilities for GPU tensors.
@internal
extension type const ShapeUtils._(Object? _) {
  /// Computes default C-contiguous strides (in element counts) for [shape].
  static List<int> computeCStrides(List<int> shape) =>
      _topComputeCStrides(shape);

  /// Calculates total number of elements in a tensor of [shape].
  static int computeSize(List<int> shape) => _topComputeSize(shape);

  /// Checks whether shapes [a] and [b] are identical in dimensions.
  static bool areEqual(List<int> a, List<int> b) => areShapesEqual(a, b);

  /// Checks whether [shape] and [strides] represent a contiguous C-layout.
  static bool isContiguous(List<int> shape, List<int> strides) =>
      isContiguousLayout(shape, strides);

  /// Broadcasts two tensor shapes according to NumPy-style broadcasting rules.
  static List<int> broadcastShapes(List<int> shapeA, List<int> shapeB) =>
      _topBroadcastShapes(shapeA, shapeB);

  /// Computes broadcasted strides for an array with [shape] and [strides]
  /// when expanded to [targetShape].
  static List<int> broadcastStrides(
    List<int> shape,
    List<int> strides,
    List<int> targetShape,
  ) => _topBroadcastStrides(shape, strides, targetShape);
}

const _topComputeCStrides = computeCStrides;
const _topComputeSize = computeSize;
const _topBroadcastShapes = broadcastShapes;
const _topBroadcastStrides = broadcastStrides;

/// Internal buffer read/write helpers for GPU tensors.
@internal
extension type const ComputeEngine._(Object? _) {
  /// Reads a single typed value at [elementIndex] from [buffer].
  static Object readAny(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex, {
    int offsetElements = 0,
  }) => readBufferAny(
    buffer,
    dtype,
    elementIndex,
    offsetElements: offsetElements,
  );

  /// Writes a single typed value at [elementIndex] to [buffer].
  static void writeAny(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex,
    Object? value, {
    int offsetElements = 0,
  }) => writeBufferAny(
    buffer,
    dtype,
    elementIndex,
    value,
    offsetElements: offsetElements,
  );

  /// Reads a single numerical value at [elementIndex] from [buffer].
  static double readValue(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex, {
    int offsetElements = 0,
  }) => readBufferValue(
    buffer,
    dtype,
    elementIndex,
    offsetElements: offsetElements,
  );

  /// Writes a single numerical value at [elementIndex] to [buffer].
  static void writeValue(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex,
    double value, {
    int offsetElements = 0,
  }) => writeBufferValue(
    buffer,
    dtype,
    elementIndex,
    value,
    offsetElements: offsetElements,
  );

  /// Encodes [value] as a 16-bit IEEE 754 float16 bit pattern.
  static int doubleToFloat16Bits(double value) =>
      Float16Utils.encodeFloat16(value);

  /// Encodes [value] as a 16-bit bfloat16 bit pattern.
  static int doubleToBfloat16Bits(double value) =>
      Float16Utils.encodeBFloat16(value);
}
