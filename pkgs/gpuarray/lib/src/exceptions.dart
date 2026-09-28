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

/// Exception thrown when a GPU array or device operation fails.
class GpuException implements Exception {
  /// The error message.
  final String message;

  /// Creates a new [GpuException] with the given [message].
  const GpuException(this.message);

  @override
  String toString() => 'GpuException: $message';
}

/// Exception thrown when an invalid buffer or out-of-memory error occurs.
class GpuMemoryException extends GpuException {
  /// Creates a new [GpuMemoryException] with the given [message].
  const GpuMemoryException(super.message);

  @override
  String toString() => 'GpuMemoryException: $message';
}

/// Exception thrown when accessing a disposed GPU resource or device.
class GpuDeviceDisposedException extends GpuException {
  /// Creates a new [GpuDeviceDisposedException] with the given [message].
  const GpuDeviceDisposedException([
    super.message = 'Attempted to access a disposed GPU resource or device.',
  ]);

  @override
  String toString() => 'GpuDeviceDisposedException: $message';
}

/// Exception thrown when tensor shapes are incompatible for an operation or broadcasting.
class GpuShapeMismatchException extends GpuException {
  /// The shape of the first operand.
  final List<int> shapeA;

  /// The shape of the second operand.
  final List<int> shapeB;

  /// The operation name.
  final String operation;

  /// Creates a new [GpuShapeMismatchException].
  GpuShapeMismatchException(this.operation, this.shapeA, this.shapeB)
    : super(
        'Cannot perform $operation on incompatible shapes: $shapeA and $shapeB',
      );

  @override
  String toString() => 'GpuShapeMismatchException: $message';
}

/// Exception thrown when an axis index is out of bounds for a tensor.
class GpuAxisOutOfBoundsException extends GpuException {
  /// The requested axis.
  final int axis;

  /// The tensor rank.
  final int rank;

  /// Creates a new [GpuAxisOutOfBoundsException].
  GpuAxisOutOfBoundsException(this.axis, this.rank)
    : super('Axis $axis is out of bounds for tensor of rank $rank');

  @override
  String toString() => 'GpuAxisOutOfBoundsException: $message';
}

/// Exception thrown when a GPU device initialization or driver operation fails.
class GpuDeviceException extends GpuException {
  /// Creates a new [GpuDeviceException] with the given [message].
  const GpuDeviceException(super.message);

  @override
  String toString() => 'GpuDeviceException: $message';
}

/// Exception thrown when a compute shader compilation or pipeline dispatch fails.
class GpuComputeException extends GpuException {
  /// Creates a new [GpuComputeException] with the given [message].
  const GpuComputeException(super.message);

  @override
  String toString() => 'GpuComputeException: $message';
}
