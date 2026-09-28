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
import 'package:resource_scope/resource_scope.dart';
import 'exceptions.dart';
import 'device.dart';
import 'backend/memory_pool.dart';

/// Usage flags for a [GpuBuffer], indicating allowed operations and access modes.
final class GpuBufferUsage {
  final int value;
  const GpuBufferUsage(this.value);

  static const GpuBufferUsage storage = GpuBufferUsage(1 << 0);
  static const GpuBufferUsage copySrc = GpuBufferUsage(1 << 1);
  static const GpuBufferUsage copyDst = GpuBufferUsage(1 << 2);
  static const GpuBufferUsage uniform = GpuBufferUsage(1 << 3);

  GpuBufferUsage operator |(GpuBufferUsage other) =>
      GpuBufferUsage(value | other.value);

  bool contains(GpuBufferUsage flag) => (value & flag.value) == flag.value;

  @override
  String toString() {
    final parts = <String>[];
    if (contains(storage)) parts.add('STORAGE');
    if (contains(copySrc)) parts.add('COPY_SRC');
    if (contains(copyDst)) parts.add('COPY_DST');
    if (contains(uniform)) parts.add('UNIFORM');
    return 'GpuBufferUsage(${parts.join("|")})';
  }
}

/// A block of memory allocated on a GPU device.
final class GpuBuffer implements ffi.Finalizable, ScopedResource {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  final ffi.Pointer<ffi.Void> _pointer;
  final int _sizeInBytes;
  GpuBufferUsage _usage;
  final GpuDevice _device;
  final GpuMemoryPool? _originPool;
  final bool _isUnmanaged;

  int _refCount = 1;
  bool _isDisposed = false;

  GpuBuffer._(
    this._pointer,
    this._sizeInBytes,
    this._usage,
    this._device, {
    GpuMemoryPool? originPool,
    bool isUnmanaged = false,
  }) : _originPool = originPool,
       _isUnmanaged = isUnmanaged {
    if (!_isUnmanaged) {
      if (_pointer != ffi.nullptr && _originPool == null) {
        _finalizer.attach(this, _pointer, detach: this);
      }
      ResourceScope.track(this);
    }
    _device.registerBuffer(this);
  }

  /// Whether this buffer wraps an unmanaged native pointer.
  bool get isUnmanaged => _isUnmanaged;

  /// Retains a reference to this buffer.
  void retain() {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot retain a disposed GpuBuffer.');
    }
    _refCount++;
  }

  /// Releases a reference to this buffer.
  void release() {
    if (_isDisposed) return;
    _refCount--;
    if (_refCount <= 0) {
      _disposeInternal();
    }
  }

  int get refCount => _refCount;

  void resetForReuse({GpuBufferUsage? usage}) {
    _refCount = 1;
    _isDisposed = false;
    _usage = usage ?? _usage;
    if (!_isUnmanaged) {
      ResourceScope.track(this);
    }
    _device.registerBuffer(this);
  }

  void _disposeInternal() {
    if (_isDisposed) return;
    _isDisposed = true;
    if (!_isUnmanaged) {
      ResourceScope.untrack(this);
    }
    _device.unregisterBuffer(this);

    if (_isUnmanaged) {
      return;
    }

    if (_originPool != null) {
      _originPool.recycle(this);
      return;
    }

    if (_pointer != ffi.nullptr) {
      _finalizer.detach(this);
      _device.backend.freeBuffer(_pointer.cast<ffi.Uint8>(), _sizeInBytes);
    }
  }

  /// Allocates a new [GpuBuffer] on the given [device].
  factory GpuBuffer.allocate({
    required int sizeInBytes,
    required GpuBufferUsage usage,
    required GpuDevice device,
  }) {
    if (sizeInBytes < 0) {
      throw ArgumentError.value(
        sizeInBytes,
        'sizeInBytes',
        'Buffer size cannot be negative.',
      );
    }
    if (device.isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot allocate buffer on disposed device.',
      );
    }

    final ptr = sizeInBytes == 0
        ? ffi.nullptr
        : device.backend.allocateBuffer(sizeInBytes).cast<ffi.Void>();

    return GpuBuffer._(ptr, sizeInBytes, usage, device);
  }

  /// Wraps an existing allocated native pointer as a [GpuBuffer] allocated from [originPool].
  factory GpuBuffer.fromPool(
    ffi.Pointer<ffi.Uint8> pointer,
    int sizeInBytes,
    GpuMemoryPool originPool, {
    required GpuDevice device,
    GpuBufferUsage usage = GpuBufferUsage.storage,
  }) {
    return GpuBuffer._(
      pointer.cast<ffi.Void>(),
      sizeInBytes,
      usage,
      device,
      originPool: originPool,
    );
  }

  /// Wraps an existing native pointer as an unmanaged [GpuBuffer].
  factory GpuBuffer.unmanaged(
    ffi.Pointer<ffi.Void> pointer,
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.storage,
    GpuDevice? device,
  }) {
    if (sizeInBytes < 0) {
      throw ArgumentError.value(
        sizeInBytes,
        'sizeInBytes',
        'Buffer size cannot be negative.',
      );
    }
    return GpuBuffer._(
      pointer,
      sizeInBytes,
      usage,
      device ?? GpuDevice.defaultDevice,
      isUnmanaged: true,
    );
  }

  ffi.Pointer<ffi.Void> get address {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot access address of disposed GpuBuffer.');
    }
    return _pointer;
  }

  /// Raw native pointer without disposed check, for internal driver use.
  ffi.Pointer<ffi.Void> get rawPointer => _pointer;

  /// Alias for [address] representing the raw native memory pointer.
  ffi.Pointer<ffi.Void> get pointer => address;

  int get sizeInBytes => _sizeInBytes;
  GpuBufferUsage get usage => _usage;
  GpuDevice get device => _device;

  @override
  bool get isDisposed => _isDisposed;

  void copyFromHost(
    ffi.Pointer<ffi.Void> srcPointer,
    int bytes, {
    int offset = 0,
  }) {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot copy to disposed GpuBuffer.');
    }
    if (bytes < 0 || offset < 0 || offset + bytes > _sizeInBytes) {
      throw GpuMemoryException('Copy operation exceeds buffer boundaries.');
    }
    if (bytes == 0) return;
    _device.backend.copyHostToBuffer(
      srcPointer.cast<ffi.Uint8>(),
      this,
      bytes,
      offset: offset,
    );
  }

  void copyToHost(
    ffi.Pointer<ffi.Void> dstPointer,
    int bytes, {
    int offset = 0,
  }) {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot copy from disposed GpuBuffer.');
    }
    if (bytes < 0 || offset < 0 || offset + bytes > _sizeInBytes) {
      throw GpuMemoryException('Copy operation exceeds buffer boundaries.');
    }
    if (bytes == 0) return;
    _device.backend.copyBufferToHost(
      this,
      dstPointer.cast<ffi.Uint8>(),
      bytes,
      offset: offset,
    );
  }

  void copyToBuffer(
    GpuBuffer dstBuffer,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    if (_isDisposed || dstBuffer.isDisposed) {
      throw GpuMemoryException('Cannot copy with disposed GpuBuffer.');
    }
    if (bytes < 0 ||
        srcOffset < 0 ||
        dstOffset < 0 ||
        srcOffset + bytes > _sizeInBytes ||
        dstOffset + bytes > dstBuffer._sizeInBytes) {
      throw GpuMemoryException('Buffer copy exceeds boundaries.');
    }
    if (bytes == 0) return;
    _device.backend.copyBufferToBuffer(
      this,
      dstBuffer,
      bytes,
      srcOffset: srcOffset,
      dstOffset: dstOffset,
    );
  }

  @override
  void dispose() => release();

  @override
  ScopedResource detachFromScope() {
    if (!_isUnmanaged) {
      ResourceScope.untrack(this);
    }
    return this;
  }

  @override
  ScopedResource detachToParentScope() {
    if (!_isUnmanaged) {
      ResourceScope.promoteToParent(this);
    }
    return this;
  }
}
