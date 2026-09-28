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
/// Supports deterministic lifecycle management via [ResourceScope] and [dispose],
/// as well as view reference counting via [retain] and [release].
final class GpuBuffer implements ffi.Finalizable, ScopedResource {
  static final ffi.NativeFinalizer _finalizer = ffi.NativeFinalizer(
    calloc.nativeFree,
  );

  final GpuDevice _device;
  final ffi.Pointer<ffi.Uint8> _address;

  /// Size of this buffer in bytes.
  final int sizeInBytes;

  GpuBufferUsage _usage;
  final bool _isPooled;
  final GpuMemoryPool? _owningPool;
  final int _allocatedBytes;

  /// Whether this buffer wraps an externally owned pointer that must not be freed by `gpuarray`.
  final bool isUnmanaged;

  bool _isDisposed = false;
  int _refCount = 1;

  GpuBuffer._({
    required GpuDevice device,
    required ffi.Pointer<ffi.Uint8> address,
    required this.sizeInBytes,
    required GpuBufferUsage usage,
    required bool isPooled,
    required int allocatedBytes,
    required this.isUnmanaged,
    GpuMemoryPool? owningPool,
  }) : _device = device,
       _address = address,
       _usage = usage,
       _isPooled = isPooled,
       _owningPool = owningPool,
       _allocatedBytes = allocatedBytes {
    if (!isUnmanaged &&
        !isPooled &&
        _address != ffi.nullptr &&
        _device.backend.usesNativeFinalizer) {
      _finalizer.attach(this, _address.cast<ffi.Void>(), detach: this);
    }
    _device.registerBuffer(this);
    if (!isUnmanaged) {
      ResourceScope.track(this);
    }
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
    final pointer = targetDevice.backend.allocateBuffer(sizeInBytes);
    return GpuBuffer._(
      device: targetDevice,
      address: pointer,
      sizeInBytes: sizeInBytes,
      usage: usage,
      isPooled: false,
      allocatedBytes: sizeInBytes,
      isUnmanaged: false,
    );
  }

  /// Internal constructor used by `GpuMemoryPool` to create pooled buffer instances.
  @internal
  factory GpuBuffer.pooled({
    required GpuDevice device,
    required ffi.Pointer<ffi.Uint8> address,
    required int sizeInBytes,
    required int allocatedBytes,
    required GpuBufferUsage usage,
    GpuMemoryPool? owningPool,
  }) {
    return GpuBuffer._(
      device: device,
      address: address,
      sizeInBytes: sizeInBytes,
      usage: usage,
      isPooled: true,
      allocatedBytes: allocatedBytes,
      isUnmanaged: false,
      owningPool: owningPool,
    );
  }

  /// Wraps an externally managed native memory [pointer] of [sizeInBytes] bytes
  /// without taking ownership of its allocation.
  ///
  /// Disposing the returned buffer will not free [pointer].
  /// It is an error if [sizeInBytes] is negative or if [device] is disposed.
  factory GpuBuffer.unmanaged(
    ffi.Pointer<ffi.Void> pointer,
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
    GpuDevice? device,
  }) {
    RangeError.checkNotNegative(sizeInBytes, 'sizeInBytes');
    final targetDevice = device ?? GpuDevice.defaultDevice;
    if (targetDevice.isDisposed) {
      throw GpuDeviceDisposedException(targetDevice.name);
    }
    return GpuBuffer._(
      device: targetDevice,
      address: pointer.cast<ffi.Uint8>(),
      sizeInBytes: sizeInBytes,
      usage: usage,
      isPooled: false,
      allocatedBytes: sizeInBytes,
      isUnmanaged: true,
    );
  }

  void _checkNotDisposed() {
    if (_isDisposed) {
      throw StateError('Cannot access a disposed GpuBuffer.');
    }
  }

  /// The [GpuDevice] on which this buffer resides.
  GpuDevice get device => _device;

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

  /// Raw native pointer without disposal checks, for internal backend cleanup only.
  @internal
  ffi.Pointer<ffi.Uint8> get rawAddress => _address;

  /// Host-accessible staging pointer for this buffer.
  ///
  /// It is an error if this buffer has been disposed.
  ffi.Pointer<ffi.Uint8> get address {
    _checkNotDisposed();
    return _address;
  }

  /// Synchronizes any pending GPU writes and returns the host-accessible
  /// native memory pointer for this buffer.
  ///
  /// It is an error if this buffer has been disposed.
  ffi.Pointer<ffi.Void> get pointer {
    _checkNotDisposed();
    _device.backend.ensureHostSynced(this);
    return _address.cast<ffi.Void>();
  }

  /// Current reference count of active tensor views sharing this buffer.
  int get refCount => _refCount;

  @override
  bool get isDisposed => _isDisposed;

  /// Ensures host staging memory is synchronized with any pending GPU writes.
  ///
  /// It is an error if this buffer has been disposed.
  void ensureHostSynced() {
    _checkNotDisposed();
    _device.backend.ensureHostSynced(this);
  }

  /// Marks this buffer's host staging memory as modified so subsequent GPU
  /// dispatches upload the updated data.
  ///
  /// It is an error if this buffer has been disposed.
  void markHostModified() {
    _checkNotDisposed();
    _device.backend.markHostModified(this);
  }

  /// Ensures the GPU buffer is synchronized with any pending host writes.
  ///
  /// It is an error if this buffer has been disposed.
  void ensureGpuSynced() {
    _checkNotDisposed();
    _device.backend.ensureGpuSynced(this);
  }

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
    _usage = newUsage;
    _device.registerBuffer(this);
    if (!isUnmanaged) {
      ResourceScope.track(this);
    }
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

  @override
  GpuBuffer detachFromScope() {
    _checkNotDisposed();
    if (!isUnmanaged) {
      ResourceScope.untrack(this);
    }
    return this;
  }

  @override
  GpuBuffer detachToParentScope() {
    _checkNotDisposed();
    if (!isUnmanaged) {
      ResourceScope.promoteToParent(this);
    }
    return this;
  }

  void _disposeInternal() {
    if (_isDisposed) return;
    _isDisposed = true;
    _refCount = 0;
    if (!isUnmanaged) {
      ResourceScope.untrack(this);
    }
    _device.unregisterBuffer(this);

    if (isUnmanaged) {
      return;
    }

    if (_isPooled) {
      final pool = _owningPool ?? _device.memoryPool;
      if (!_device.isDisposed && !pool.isDisposed) {
        pool.release(this);
      } else if (_address != ffi.nullptr) {
        _device.backend.freeBuffer(_address, _allocatedBytes);
      }
      return;
    }

    if (_address != ffi.nullptr) {
      if (_device.backend.usesNativeFinalizer) {
        _finalizer.detach(this);
      }
      _device.backend.freeBuffer(_address, _allocatedBytes);
    }
  }

  /// Marks this buffer disposed and frees its native memory during device shutdown.
  @internal
  void forceDisposeOnDeviceShutdown() {
    if (_isDisposed) return;
    _isDisposed = true;
    _refCount = 0;
    if (!isUnmanaged) {
      ResourceScope.untrack(this);
    }
    if (!isUnmanaged && _address != ffi.nullptr) {
      if (!_isPooled && _device.backend.usesNativeFinalizer) {
        _finalizer.detach(this);
      }
      _device.backend.freeBuffer(_address, _allocatedBytes);
    }
  }

  @override
  void dispose() {
    release();
  }
}
