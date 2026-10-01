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
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:meta/meta.dart';
import 'package:resource_scope/resource_scope.dart';

import 'backend/memory_pool.dart';
import 'device.dart';
import 'exceptions.dart';

/// Bitmask flags specifying how a [GpuBuffer] may be used in compute and transfer pipelines.
extension type const GpuBufferUsage(int mask) implements int {
  /// Buffer can be mapped for reading on the host CPU.
  static const GpuBufferUsage mapRead = GpuBufferUsage(0x0001);

  /// Buffer can be mapped for writing on the host CPU.
  static const GpuBufferUsage mapWrite = GpuBufferUsage(0x0002);

  /// Buffer can be used as the source of a copy operation.
  static const GpuBufferUsage copySrc = GpuBufferUsage(0x0004);

  /// Buffer can be used as the destination of a copy or write operation.
  static const GpuBufferUsage copyDst = GpuBufferUsage(0x0008);

  /// Buffer can be bound as a uniform buffer in a shader.
  static const GpuBufferUsage uniform = GpuBufferUsage(0x0040);

  /// Buffer can be bound as a storage buffer in a compute shader.
  static const GpuBufferUsage storage = GpuBufferUsage(0x0080);

  /// Default usage flags for general-purpose GPU compute buffers.
  static const GpuBufferUsage defaultCompute = GpuBufferUsage(
    0x0080 | 0x0004 | 0x0008,
  );

  /// Underlying integer bitmask value.
  int get value => mask;

  /// Combines this usage mask with [other] using bitwise OR.
  GpuBufferUsage operator |(GpuBufferUsage other) =>
      GpuBufferUsage(mask | other.mask);

  /// Intersects this usage mask with [other] using bitwise AND.
  GpuBufferUsage operator &(GpuBufferUsage other) =>
      GpuBufferUsage(mask & other.mask);

  /// Whether this usage mask includes all flags in [flag].
  bool contains(GpuBufferUsage flag) => (mask & flag.mask) == flag.mask;
}

/// Reference-counted GPU memory buffer associated with a [GpuDevice].
///
/// Wraps a hardware `WGPUBuffer` handle with deterministic lifecycle management
/// via [ResourceScope] and [dispose], as well as view reference counting via
/// [retain] and [release].
final class GpuBuffer implements ScopedResource {
  final GpuDevice _device;
  final ffi.Pointer<ffi.Void> _nativeHandle;
  int _sizeInBytes;
  GpuBufferUsage _usage;
  final bool _isPooled;
  final GpuMemoryPool? _owningPool;
  final int _allocatedBytes;

  bool _isDisposed = false;
  int _refCount = 1;

  GpuBuffer._({
    required GpuDevice device,
    required ffi.Pointer<ffi.Void> nativeHandle,
    required int sizeInBytes,
    required GpuBufferUsage usage,
    required bool isPooled,
    required int allocatedBytes,
    GpuMemoryPool? owningPool,
  }) : _device = device,
       _nativeHandle = nativeHandle,
       _sizeInBytes = sizeInBytes,
       _usage = usage,
       _isPooled = isPooled,
       _owningPool = owningPool,
       _allocatedBytes = allocatedBytes {
    _device.registerBuffer(this);
    ResourceScope.track(this);
  }

  /// Allocates a new [GpuBuffer] of [sizeInBytes] bytes on [device].
  ///
  /// It is an error if [sizeInBytes] is negative or if [device] has been disposed.
  factory GpuBuffer.allocate({
    required int sizeInBytes,
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
    GpuDevice? device,
  }) {
    RangeError.checkNotNegative(sizeInBytes, 'sizeInBytes');
    final targetDevice = device ?? GpuDevice.defaultDevice;
    if (targetDevice.isDisposed) {
      throw GpuDeviceDisposedException(targetDevice.name);
    }
    if (targetDevice.enableMemoryPool) {
      return targetDevice.memoryPool.acquire(sizeInBytes, usage: usage);
    }
    final alignedAllocatedBytes = sizeInBytes == 0 ? 0 : (sizeInBytes + 3) & ~3;
    final handle = sizeInBytes == 0
        ? ffi.nullptr
        : targetDevice.backend.allocateBuffer(
            alignedAllocatedBytes,
            usage: usage,
          );
    return GpuBuffer._(
      device: targetDevice,
      nativeHandle: handle,
      sizeInBytes: sizeInBytes,
      usage: usage,
      isPooled: false,
      allocatedBytes: alignedAllocatedBytes,
    );
  }

  /// Internal constructor used by `GpuMemoryPool` to create pooled buffer instances.
  @internal
  factory GpuBuffer.pooled({
    required GpuDevice device,
    required ffi.Pointer<ffi.Void> nativeHandle,
    required int sizeInBytes,
    required int allocatedBytes,
    required GpuBufferUsage usage,
    GpuMemoryPool? owningPool,
  }) {
    return GpuBuffer._(
      device: device,
      nativeHandle: nativeHandle,
      sizeInBytes: sizeInBytes,
      usage: usage,
      isPooled: true,
      allocatedBytes: allocatedBytes,
      owningPool: owningPool,
    );
  }

  void _checkNotDisposed() {
    if (_isDisposed) {
      throw StateError('Cannot access a disposed GpuBuffer.');
    }
  }

  /// The [GpuDevice] on which this buffer resides.
  GpuDevice get device => _device;

  /// Logical size of this buffer in bytes.
  int get sizeInBytes => _sizeInBytes;

  /// Usage flags configured for this buffer.
  GpuBufferUsage get usage => _usage;

  /// Updates the usage flags when this buffer is recycled from a memory pool.
  @internal
  set usage(GpuBufferUsage value) {
    _usage = value;
  }

  /// Physical size in bytes allocated for this buffer (including pool bucket rounding).
  @internal
  int get allocatedBytes => _allocatedBytes;

  /// Native `WGPUBuffer` handle without disposal checks, for internal backend cleanup only.
  @internal
  ffi.Pointer<ffi.Void> get rawNativeHandle => _nativeHandle;

  /// Native `WGPUBuffer` handle for this buffer.
  ///
  /// It is an error if this buffer has been disposed.
  @internal
  ffi.Pointer<ffi.Void> get nativeHandle {
    _checkNotDisposed();
    return _nativeHandle;
  }

  /// Current reference count of active tensor views sharing this buffer.
  int get refCount => _refCount;

  @override
  bool get isDisposed => _isDisposed;

  /// Increments the reference count when a new view is created over this buffer.
  ///
  /// It is an error if this buffer has been disposed.
  void retain() {
    _checkNotDisposed();
    _refCount++;
  }

  /// Decrements the reference count and disposes the underlying memory when
  /// the count reaches zero.
  void release() {
    if (_isDisposed) return;
    _refCount--;
    if (_refCount <= 0) {
      _disposeInternal();
    }
  }

  /// Re-initializes a pooled buffer when re-acquired from `GpuMemoryPool`.
  @internal
  void reviveFromPool({
    required int requestedSize,
    required GpuBufferUsage newUsage,
  }) {
    _isDisposed = false;
    _refCount = 1;
    _sizeInBytes = requestedSize;
    _usage = newUsage;
    _device.registerBuffer(this);
    ResourceScope.track(this);
  }

  /// Zero-fills [bytes] (or [length]) bytes of this buffer starting at byte [offset] on the GPU.
  ///
  /// If [bytes] and [length] are omitted, clears from [offset] to the end of the buffer.
  /// It is an error if this buffer has been disposed.
  /// Throws a [GpuMemoryException] if the range is out of bounds.
  void clear({int offset = 0, int? bytes, int? length}) {
    _checkNotDisposed();
    _device.backend.clearBuffer(this, offset: offset, bytes: bytes ?? length);
  }

  /// Copies [bytes] bytes from [hostPtr] into this buffer at byte [offset].
  ///
  /// It is an error if this buffer has been disposed.
  /// Throws a [GpuMemoryException] if the copy range is out of bounds.
  void copyFromHost(
    ffi.Pointer<ffi.Void> hostPtr,
    int bytes, {
    int offset = 0,
  }) {
    _checkNotDisposed();
    _device.backend.copyHostToBuffer(
      hostPtr.cast<ffi.Uint8>(),
      this,
      bytes,
      offset: offset,
    );
  }

  /// Copies [bytes] bytes from this buffer at byte [offset] into [hostPtr].
  ///
  /// It is an error if this buffer has been disposed.
  /// Throws a [GpuMemoryException] if the copy range is out of bounds.
  void copyToHost(ffi.Pointer<ffi.Void> hostPtr, int bytes, {int offset = 0}) {
    _checkNotDisposed();
    _device.backend.copyBufferToHost(
      this,
      hostPtr.cast<ffi.Uint8>(),
      bytes,
      offset: offset,
    );
  }

  /// Writes [data] into this buffer starting at byte [offset].
  ///
  /// It is an error if this buffer has been disposed.
  /// Throws a [GpuMemoryException] if the write range exceeds [sizeInBytes].
  void writeBytes(List<int> data, {int offset = 0}) {
    _checkNotDisposed();
    if (offset < 0 || offset + data.length > _sizeInBytes) {
      throw GpuMemoryException(
        'Copy bounds (offset: $offset, bytes: ${data.length}) exceed destination buffer size ($_sizeInBytes).',
      );
    }
    if (data.isEmpty) return;
    using((arena) {
      final hostStaging = arena<ffi.Uint8>(data.length);
      hostStaging.asTypedList(data.length).setAll(0, data);
      _device.backend.copyHostToBuffer(
        hostStaging,
        this,
        data.length,
        offset: offset,
      );
    });
  }

  /// Reads [bytes] (or [length]) bytes (or all remaining bytes from [offset]) from this buffer into a [Uint8List].
  ///
  /// It is an error if this buffer has been disposed.
  /// Throws a [GpuMemoryException] if the read range exceeds [sizeInBytes].
  Uint8List readBytes({int offset = 0, int? bytes, int? length}) {
    _checkNotDisposed();
    final readLength = bytes ?? length ?? (_sizeInBytes - offset);
    if (offset < 0 || readLength < 0 || offset + readLength > _sizeInBytes) {
      throw GpuMemoryException(
        'Copy bounds (offset: $offset, bytes: $readLength) exceed source buffer size ($_sizeInBytes).',
      );
    }
    final result = Uint8List(readLength);
    if (readLength == 0) return result;
    using((arena) {
      final hostStaging = arena<ffi.Uint8>(readLength);
      _device.backend.copyBufferToHost(
        this,
        hostStaging,
        readLength,
        offset: offset,
      );
      result.setAll(0, hostStaging.asTypedList(readLength));
    });
    return result;
  }

  /// Copies [bytes] bytes from this buffer (at [srcOffset]) to [dst] (at [dstOffset]).
  ///
  /// It is an error if this buffer or [dst] has been disposed.
  /// Throws a [GpuMemoryException] if the copy range is out of bounds.
  void copyToBuffer(
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    _checkNotDisposed();
    if (dst.isDisposed) {
      throw StateError('Cannot copy into a disposed destination GpuBuffer.');
    }
    _device.backend.copyBufferToBuffer(
      this,
      dst,
      bytes,
      srcOffset: srcOffset,
      dstOffset: dstOffset,
    );
  }

  /// Copies [size] (or [length] / [bytes]) bytes from [sourceOffset] (or [srcOffset]) to [destination] at [destinationOffset] (or [dstOffset]).
  ///
  /// It is an error if this buffer or [destination] has been disposed.
  /// Throws a [GpuMemoryException] if the copy range is out of bounds.
  void copyTo(
    GpuBuffer destination, {
    int sourceOffset = 0,
    int destinationOffset = 0,
    int? srcOffset,
    int? dstOffset,
    int? size,
    int? bytes,
    int? length,
  }) {
    _checkNotDisposed();
    final effectiveSrcOffset = srcOffset ?? sourceOffset;
    final effectiveDstOffset = dstOffset ?? destinationOffset;
    final copyLength =
        size ?? bytes ?? length ?? (_sizeInBytes - effectiveSrcOffset);
    copyToBuffer(
      destination,
      copyLength,
      srcOffset: effectiveSrcOffset,
      dstOffset: effectiveDstOffset,
    );
  }

  @override
  GpuBuffer detachFromScope() {
    _checkNotDisposed();
    ResourceScope.untrack(this);
    return this;
  }

  @override
  GpuBuffer detachToParentScope() {
    _checkNotDisposed();
    ResourceScope.promoteToParent(this);
    return this;
  }

  void _disposeInternal() {
    if (_isDisposed) return;
    _isDisposed = true;
    _refCount = 0;
    ResourceScope.untrack(this);
    _device.unregisterBuffer(this);

    if (_isPooled) {
      final pool = _owningPool ?? _device.memoryPool;
      if (!_device.isDisposed && !pool.isDisposed) {
        pool.release(this);
      } else if (_nativeHandle != ffi.nullptr) {
        _device.backend.freeBuffer(_nativeHandle, _allocatedBytes);
      }
      return;
    }

    if (_nativeHandle != ffi.nullptr) {
      _device.backend.freeBuffer(_nativeHandle, _allocatedBytes);
    }
  }

  /// Marks this buffer disposed and frees its native memory during device shutdown.
  @internal
  void forceDisposeOnDeviceShutdown() {
    if (_isDisposed) return;
    _isDisposed = true;
    _refCount = 0;
    ResourceScope.untrack(this);
    if (_nativeHandle != ffi.nullptr) {
      _device.backend.freeBuffer(_nativeHandle, _allocatedBytes);
    }
  }

  @override
  void dispose() {
    release();
  }
}
