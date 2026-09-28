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

import 'package:ffi/ffi.dart';

import '../buffer.dart';
import '../device.dart';
import '../exceptions.dart';
import 'wgsl/wgsl_types.dart';

export '../device.dart' show GpuDeviceType;

/// Abstract hardware execution and memory driver backend for a [GpuDevice].
///
/// Implementations manage device memory allocation, host-device transfers,
/// and compute kernel dispatch (e.g., native WebGPU via `wgpu-native`, browser
/// WebGPU via `dart:js_interop`, or host SIMD via [CpuVectorBackend]).
abstract class GpuBackend {
  /// Creates a [GpuBackend].
  const GpuBackend();

  /// The hardware device type managed by this backend.
  GpuDeviceType get deviceType;

  /// Whether this backend executes in a simulated or host CPU fallback mode
  /// rather than dispatching to a physical hardware GPU driver.
  bool get isSimulated => true;

  /// Whether [GpuBuffer] instances on this backend should attach a
  /// `calloc.nativeFree` [ffi.NativeFinalizer] as a garbage-collection safety net.
  bool get usesNativeFinalizer => false;

  /// Allocates a raw buffer of [sizeInBytes] bytes.
  ///
  /// Throws a [GpuMemoryException] if the allocation fails.
  ffi.Pointer<ffi.Uint8> allocateBuffer(int sizeInBytes);

  /// Frees a raw buffer previously allocated by [allocateBuffer].
  void freeBuffer(ffi.Pointer<ffi.Uint8> pointer, int sizeInBytes);

  /// Ensures the host-accessible staging memory for [buffer] reflects the
  /// latest GPU buffer state before host CPU reads.
  void ensureHostSynced(GpuBuffer buffer) {}

  /// Marks the host-accessible memory for [buffer] as modified by the host CPU
  /// so subsequent GPU dispatches upload the updated contents.
  void markHostModified(GpuBuffer buffer) {}

  /// Ensures the GPU buffer reflects any pending host modifications before a
  /// GPU compute dispatch or device-to-device copy.
  void ensureGpuSynced(GpuBuffer buffer) {}

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
    if (bytes == 0) return;
    ensureHostSynced(dst);
    final destinationPointer = dst.rawAddress + offset;
    destinationPointer
        .asTypedList(bytes)
        .setRange(0, bytes, src.asTypedList(bytes));
    markHostModified(dst);
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
    if (bytes == 0) return;
    ensureHostSynced(src);
    final sourcePointer = src.rawAddress + offset;
    dst.asTypedList(bytes).setRange(0, bytes, sourcePointer.asTypedList(bytes));
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
    if (bytes == 0) return;
    ensureHostSynced(src);
    if (dstOffset > 0 || bytes < dst.sizeInBytes) {
      ensureHostSynced(dst);
    }
    final sourcePointer = src.rawAddress + srcOffset;
    final destinationPointer = dst.rawAddress + dstOffset;
    destinationPointer
        .asTypedList(bytes)
        .setRange(0, bytes, sourcePointer.asTypedList(bytes));
    markHostModified(dst);
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
    final cpuKernel = shaderModule.metadata['cpu_kernel'];
    if (cpuKernel is Function) {
      cpuKernel(buffers, uniforms, workgroupsX, workgroupsY, workgroupsZ);
    }
  }

  /// Releases any driver resources held by this backend.
  void dispose() {}
}

/// Host CPU vector execution backend backed by zero-initialized native C heap memory.
final class CpuVectorBackend extends GpuBackend {
  /// Creates a [CpuVectorBackend].
  const CpuVectorBackend();

  @override
  GpuDeviceType get deviceType => GpuDeviceType.cpu;

  @override
  bool get usesNativeFinalizer => true;

  @override
  ffi.Pointer<ffi.Uint8> allocateBuffer(int sizeInBytes) {
    if (sizeInBytes < 0) {
      throw ArgumentError.value(
        sizeInBytes,
        'sizeInBytes',
        'Must be non-negative.',
      );
    }
    if (sizeInBytes == 0) {
      return ffi.nullptr;
    }
    final pointer = calloc<ffi.Uint8>(sizeInBytes);
    if (pointer == ffi.nullptr) {
      throw GpuMemoryException(
        'Failed to allocate $sizeInBytes bytes on host heap.',
      );
    }
    return pointer;
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Uint8> pointer, int sizeInBytes) {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}
