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
import '../exceptions.dart';
import 'wgsl/wgsl_types.dart';

/// Device driver classification for hardware compute acceleration.
enum GpuDeviceType {
  /// CPU Vector / SIMD backend via native FFI C pointers.
  cpu,

  /// WebGPU / wgpu hardware compute backend via WGSL shaders.
  webgpu,
}

/// Abstract hardware driver backend for memory allocation and compute dispatch.
abstract class GpuBackend {
  const GpuBackend();

  /// The type of hardware device targeted by this backend.
  GpuDeviceType get deviceType;

  /// Whether this backend runs in software/CPU simulation mode.
  bool get isSimulated => true;

  /// Allocates a low-level memory buffer of [sizeInBytes].
  ffi.Pointer<ffi.Uint8> allocateBuffer(int sizeInBytes);

  /// Frees a low-level memory buffer.
  void freeBuffer(ffi.Pointer<ffi.Uint8> pointer, int sizeInBytes);

  /// Copies [bytes] from host memory [src] to device buffer [dst].
  void copyHostToBuffer(
    ffi.Pointer<ffi.Uint8> src,
    GpuBuffer dst,
    int bytes, {
    int offset = 0,
  }) {
    if (dst.isDisposed) {
      throw GpuMemoryException('Cannot copy to disposed GpuBuffer.');
    }
    if (bytes < 0 || offset < 0 || offset + bytes > dst.sizeInBytes) {
      throw GpuMemoryException('Copy operation exceeds buffer boundaries.');
    }
    if (bytes == 0) return;
    final dstPtr = dst.address.cast<ffi.Uint8>();
    final srcBytes = src.asTypedList(bytes);
    final dstBytes = (dstPtr + offset).asTypedList(bytes);
    dstBytes.setAll(0, srcBytes);
  }

  /// Copies [bytes] from device buffer [src] to host memory [dst].
  void copyBufferToHost(
    GpuBuffer src,
    ffi.Pointer<ffi.Uint8> dst,
    int bytes, {
    int offset = 0,
  }) {
    if (src.isDisposed) {
      throw GpuMemoryException('Cannot copy from disposed GpuBuffer.');
    }
    if (bytes < 0 || offset < 0 || offset + bytes > src.sizeInBytes) {
      throw GpuMemoryException('Copy operation exceeds buffer boundaries.');
    }
    if (bytes == 0) return;
    final srcPtr = src.address.cast<ffi.Uint8>();
    final srcBytes = (srcPtr + offset).asTypedList(bytes);
    final dstBytes = dst.asTypedList(bytes);
    dstBytes.setAll(0, srcBytes);
  }

  /// Copies [bytes] between two device buffers.
  void copyBufferToBuffer(
    GpuBuffer src,
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    if (src.isDisposed || dst.isDisposed) {
      throw GpuMemoryException('Cannot copy with disposed GpuBuffer.');
    }
    if (bytes < 0 ||
        srcOffset < 0 ||
        dstOffset < 0 ||
        srcOffset + bytes > src.sizeInBytes ||
        dstOffset + bytes > dst.sizeInBytes) {
      throw GpuMemoryException('Buffer copy exceeds boundaries.');
    }
    if (bytes == 0) return;
    final srcPtr = src.address.cast<ffi.Uint8>();
    final dstPtr = dst.address.cast<ffi.Uint8>();
    final srcBytes = (srcPtr + srcOffset).asTypedList(bytes);
    final dstBytes = (dstPtr + dstOffset).asTypedList(bytes);
    dstBytes.setAll(0, srcBytes);
  }

  /// Dispatches a WGSL compute shader pipeline over the provided [buffers].
  void dispatchComputePipeline({
    required WgslShaderModule shaderModule,
    required List<GpuBuffer> buffers,
    List<int>? uniforms,
    required int workgroupsX,
    int workgroupsY = 1,
    int workgroupsZ = 1,
  }) {
    // Default base implementation for CPU/fallback backends.
  }
}

/// Default CPU Vector & SIMD memory driver backend using calloc native allocations.
class CpuVectorBackend extends GpuBackend {
  const CpuVectorBackend();

  @override
  GpuDeviceType get deviceType => GpuDeviceType.cpu;

  @override
  ffi.Pointer<ffi.Uint8> allocateBuffer(int sizeInBytes) {
    if (sizeInBytes <= 0) return ffi.nullptr;
    return calloc<ffi.Uint8>(sizeInBytes);
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Uint8> pointer, int sizeInBytes) {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }

  @override
  void dispatchComputePipeline({
    required WgslShaderModule shaderModule,
    required List<GpuBuffer> buffers,
    List<int>? uniforms,
    required int workgroupsX,
    int workgroupsY = 1,
    int workgroupsZ = 1,
  }) {
    if (buffers.any((b) => b.isDisposed)) {
      throw GpuMemoryException(
        'Cannot dispatch pipeline with disposed buffers.',
      );
    }
    final cpuKernel = shaderModule.metadata['cpu_kernel'];
    if (cpuKernel is Function) {
      cpuKernel(buffers, uniforms, workgroupsX, workgroupsY, workgroupsZ);
    }
  }
}
