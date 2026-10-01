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

import '../buffer.dart';
import '../device.dart';
import '../exceptions.dart';
import 'wgsl/wgsl_types.dart';

export '../device.dart' show GpuDeviceType;

/// Abstract hardware execution and memory driver backend for a [GpuDevice].
///
/// Implementations manage device memory allocation, host-device transfers,
/// and compute kernel dispatch (such as native WebGPU via `wgpu-native` or
/// browser WebGPU via `dart:js_interop`).
abstract class GpuBackend {
  /// Creates a [GpuBackend].
  const GpuBackend();

  /// The hardware device type managed by this backend.
  GpuDeviceType get deviceType;

  /// Whether buffers allocated by this backend attach a Dart `NativeFinalizer`.
  bool get usesNativeFinalizer => false;

  /// Allocates a device buffer handle of [sizeInBytes] bytes with [usage] flags.
  ///
  /// Throws a [GpuMemoryException] if the allocation fails.
  ffi.Pointer<ffi.Void> allocateBuffer(
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
  });

  /// Frees a device buffer handle previously allocated by [allocateBuffer].
  void freeBuffer(ffi.Pointer<ffi.Void> handle, int sizeInBytes);

  /// Zero-fills [bytes] bytes of [buffer] starting at byte [offset].
  ///
  /// If [bytes] is omitted, clears from [offset] to [GpuBuffer.sizeInBytes].
  /// Throws a [GpuMemoryException] if [buffer] is disposed or the range is out of bounds.
  void clearBuffer(GpuBuffer buffer, {int offset = 0, int? bytes}) {
    if (buffer.isDisposed) {
      throw const GpuMemoryException('Cannot clear a disposed GpuBuffer.');
    }
    final resolvedBytes = bytes ?? (buffer.sizeInBytes - offset);
    if (offset < 0 ||
        resolvedBytes < 0 ||
        offset + resolvedBytes > buffer.allocatedBytes) {
      throw GpuMemoryException(
        'Clear bounds (offset: $offset, bytes: $resolvedBytes) exceed buffer size (${buffer.allocatedBytes}).',
      );
    }
  }

  /// Copies [bytes] bytes from [src] into [dst] starting at byte [offset].
  ///
  /// Throws a [GpuMemoryException] if [bytes] or [offset] is negative or
  /// exceeds [dst]'s size.
  void copyHostToBuffer(
    ffi.Pointer<ffi.Uint8> src,
    GpuBuffer dst,
    int bytes, {
    int offset = 0,
  }) {
    if (dst.isDisposed) {
      throw const GpuMemoryException('Cannot copy into a disposed GpuBuffer.');
    }
    if (bytes < 0 || offset < 0 || offset + bytes > dst.sizeInBytes) {
      throw GpuMemoryException(
        'Copy bounds (offset: $offset, bytes: $bytes) exceed destination buffer size (${dst.sizeInBytes}).',
      );
    }
  }

  /// Copies [bytes] bytes from [src] starting at byte [offset] into [dst].
  ///
  /// Throws a [GpuMemoryException] if [bytes] or [offset] is negative or
  /// exceeds [src]'s size.
  void copyBufferToHost(
    GpuBuffer src,
    ffi.Pointer<ffi.Uint8> dst,
    int bytes, {
    int offset = 0,
  }) {
    if (src.isDisposed) {
      throw const GpuMemoryException('Cannot copy from a disposed GpuBuffer.');
    }
    if (bytes < 0 || offset < 0 || offset + bytes > src.sizeInBytes) {
      throw GpuMemoryException(
        'Copy bounds (offset: $offset, bytes: $bytes) exceed source buffer size (${src.sizeInBytes}).',
      );
    }
  }

  /// Copies [bytes] bytes from [src] (at [srcOffset]) to [dst] (at [dstOffset]).
  ///
  /// Throws a [GpuMemoryException] if any offset or byte count is negative or
  /// exceeds buffer bounds.
  void copyBufferToBuffer(
    GpuBuffer src,
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    if (src.isDisposed || dst.isDisposed) {
      throw const GpuMemoryException(
        'Cannot copy between disposed GpuBuffers.',
      );
    }
    if (bytes < 0 ||
        srcOffset < 0 ||
        dstOffset < 0 ||
        srcOffset + bytes > src.sizeInBytes ||
        dstOffset + bytes > dst.sizeInBytes) {
      throw GpuMemoryException(
        'Buffer-to-buffer copy bounds (srcOffset: $srcOffset, dstOffset: $dstOffset, bytes: $bytes) '
        'exceed source (${src.sizeInBytes}) or destination (${dst.sizeInBytes}) size.',
      );
    }
  }

  /// Dispatches a compiled [shaderModule] compute pipeline over [buffers] and
  /// optional 32-bit [uniforms] across a `(workgroupsX, workgroupsY, workgroupsZ)` grid.
  ///
  /// It is an error if [workgroupsX], [workgroupsY], or [workgroupsZ] is less than `1`.
  /// Throws a [GpuMemoryException] if any buffer in [buffers] is disposed.
  void dispatchComputePipeline({
    required WgslShaderModule shaderModule,
    required List<GpuBuffer> buffers,
    List<int>? uniforms,
    required int workgroupsX,
    int workgroupsY = 1,
    int workgroupsZ = 1,
  }) {
    if (workgroupsX <= 0 || workgroupsY <= 0 || workgroupsZ <= 0) {
      throw ArgumentError.value(
        (workgroupsX, workgroupsY, workgroupsZ),
        'workgroups',
        'Must have positive workgroup dimensions (workgroupsX, workgroupsY, workgroupsZ >= 1).',
      );
    }
    for (var i = 0; i < buffers.length; i++) {
      if (buffers[i].isDisposed) {
        throw GpuMemoryException(
          'Cannot dispatch compute pipeline "${shaderModule.name}": buffer at binding $i is disposed.',
        );
      }
    }
  }

  /// Releases any driver resources held by this backend.
  void dispose() {}
}
