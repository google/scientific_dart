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

import 'dtype.dart';

/// Base interface for errors and exceptions raised by `package:gpuarray`.
abstract interface class GpuException implements Exception {
  /// Human-readable description of the failure.
  String get message;
}

/// Exception thrown when a GPU memory allocation or buffer transfer fails.
final class GpuMemoryException implements GpuException {
  @override
  final String message;

  /// Creates a [GpuMemoryException] with the given [message].
  const GpuMemoryException(this.message);

  @override
  String toString() => 'GpuMemoryException: $message';
}

/// Error thrown when an operation is attempted on a disposed [GpuDevice] or [GpuBuffer].
final class GpuDeviceDisposedException extends StateError
    implements GpuException {
  /// Creates a [GpuDeviceDisposedException] with the given [message].
  GpuDeviceDisposedException([
    super.message = 'Attempted to access a disposed GPU resource or device.',
  ]);

  @override
  String toString() => 'GpuDeviceDisposedException: $message';
}

/// Error thrown when tensor shapes are incompatible for an operation or broadcasting.
final class GpuShapeMismatchException extends ArgumentError
    implements GpuException {
  /// The shape of the first operand.
  final List<int> shapeA;

  /// The shape of the second operand.
  final List<int> shapeB;

  /// The operation name.
  final String operation;

  /// Creates a [GpuShapeMismatchException] with unmodifiable copies of [shapeA] and [shapeB].
  GpuShapeMismatchException(this.operation, List<int> shapeA, List<int> shapeB)
    : shapeA = List<int>.unmodifiable(shapeA),
      shapeB = List<int>.unmodifiable(shapeB),
      super(
        'Cannot perform $operation on incompatible shapes: $shapeA and $shapeB',
      );

  @override
  String get message => super.message?.toString() ?? '';

  @override
  String toString() => 'GpuShapeMismatchException: $message';
}

/// Error thrown when an axis index is out of bounds for a tensor of a given rank.
final class GpuAxisOutOfBoundsException extends RangeError
    implements GpuException {
  /// The requested axis.
  final int axis;

  /// The tensor rank.
  final int rank;

  /// Creates a [GpuAxisOutOfBoundsException].
  GpuAxisOutOfBoundsException(this.axis, this.rank)
    : super('Axis $axis is out of bounds for tensor of rank $rank');

  @override
  String get message => super.message?.toString() ?? '';

  @override
  String toString() => 'GpuAxisOutOfBoundsException: $message';
}

/// Error thrown when an operation does not support a given [DType].
final class GpuUnsupportedDTypeException extends UnsupportedError
    implements GpuException {
  /// The unsupported data type.
  final DType dtype;

  /// The operation name.
  final String operation;

  /// Creates a [GpuUnsupportedDTypeException].
  GpuUnsupportedDTypeException(this.operation, this.dtype)
    : super('Operation "$operation" does not support dtype ${dtype.name}.');

  @override
  String get message => super.message ?? '';

  @override
  String toString() => 'GpuUnsupportedDTypeException: $message';
}

/// Exception thrown when a GPU device initialization or driver operation fails.
final class GpuDeviceException implements GpuException {
  @override
  final String message;

  /// Creates a [GpuDeviceException] with the given [message].
  const GpuDeviceException(this.message);

  @override
  String toString() => 'GpuDeviceException: $message';
}

/// Exception thrown when a compute shader compilation or pipeline dispatch fails.
final class GpuComputeException implements GpuException {
  @override
  final String message;

  /// Creates a [GpuComputeException] with the given [message].
  const GpuComputeException(this.message);

  @override
  String toString() => 'GpuComputeException: $message';
}

/// Alias for [GpuComputeException] representing shader compilation failures.
typedef GpuCompilationException = GpuComputeException;
