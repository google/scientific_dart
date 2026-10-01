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

import 'package:meta/meta.dart';
import 'package:resource_scope/resource_scope.dart';

import 'backend/backend.dart';
import 'backend/memory_pool.dart';
import 'backend/webgpu_backend.dart' show createDefaultGpuBackend;
import 'backend/wgsl/jit_compiler.dart';
import 'buffer.dart';
import 'exceptions.dart';

/// Hardware backend architecture for a [GpuDevice].
enum GpuDeviceType {
  /// Cross-platform WebGPU hardware compute backend (Metal, Vulkan, DX12, Dawn, wgpu-native).
  webgpu('WebGPU'),

  /// Apple Metal compute backend.
  metal('Metal'),

  /// Khronos Vulkan compute backend.
  vulkan('Vulkan'),

  /// NVIDIA CUDA compute backend.
  cuda('CUDA');

  /// Human-readable display label for this device type.
  final String label;

  const GpuDeviceType(this.label);
}

/// Represents a physical or virtual GPU compute device, its memory pool, and
/// its WGSL JIT kernel cache.
final class GpuDevice implements ScopedResource {
  static GpuDevice? _defaultDevice;

  /// Human-readable name of this device.
  final String name;

  /// Hardware backend type of this device.
  final GpuDeviceType type;

  /// Underlying execution and memory driver backend.
  final GpuBackend backend;

  /// Whether power-of-two bucket memory pooling is enabled on this device.
  bool enableMemoryPool;

  late final GpuMemoryPool _memoryPool;
  final WgslJitCompiler _jitCompiler = WgslJitCompiler();

  final List<WeakReference<GpuBuffer>> _activeBuffers = [];
  bool _isDisposed = false;

  GpuDevice._({
    required this.name,
    required this.type,
    required this.backend,
    required this.enableMemoryPool,
    bool trackInScope = true,
  }) {
    _memoryPool = GpuMemoryPool(this);
    if (trackInScope) {
      ResourceScope.track(this);
    }
  }

  /// Creates a new [GpuDevice] with the given [name], [type], [backend], and
  /// [enableMemoryPool] configuration.
  factory GpuDevice.create({
    String name = 'Default WebGPU Device',
    GpuDeviceType? type,
    GpuBackend? backend,
    bool enableMemoryPool = false,
  }) {
    final resolvedBackend = backend ?? createDefaultGpuBackend();
    final resolvedType = type ?? resolvedBackend.deviceType;
    return GpuDevice._(
      name: name,
      type: resolvedType,
      backend: resolvedBackend,
      enableMemoryPool: enableMemoryPool,
      trackInScope: true,
    );
  }

  /// The process-wide default [GpuDevice], lazily initialized if not yet set.
  static GpuDevice get defaultDevice {
    final current = _defaultDevice;
    if (current == null || current.isDisposed) {
      final resolvedBackend = createDefaultGpuBackend();
      final created = ResourceScope.unmanaged(
        () => GpuDevice._(
          name: 'Default GPU Device',
          type: resolvedBackend.deviceType,
          backend: resolvedBackend,
          enableMemoryPool: false,
          trackInScope: false,
        ),
      );
      _defaultDevice = created;
      return created;
    }
    return current;
  }

  /// Sets the process-wide default [GpuDevice].
  ///
  /// It is an error if [device] has been disposed.
  static set defaultDevice(GpuDevice device) {
    if (device.isDisposed) {
      throw GpuDeviceDisposedException(device.name);
    }
    _defaultDevice = device;
  }

  void _checkNotDisposed() {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(name);
    }
  }

  /// Whether this device has been disposed.
  @override
  bool get isDisposed => _isDisposed;

  /// The VRAM block memory pool associated with this device.
  GpuMemoryPool get memoryPool => _memoryPool;

  /// The WGSL JIT shader compiler and LRU pipeline cache for this device.
  WgslJitCompiler get jitCompiler {
    _checkNotDisposed();
    return _jitCompiler;
  }

  /// Total bytes currently allocated in active (non-disposed) buffers on this device.
  int get allocatedMemoryBytes {
    _pruneDeadReferences();
    var sum = 0;
    for (final ref in _activeBuffers) {
      final buffer = ref.target;
      if (buffer != null && !buffer.isDisposed) {
        sum += buffer.sizeInBytes;
      }
    }
    return sum;
  }

  /// Number of currently active (non-disposed) buffers registered on this device.
  int get activeBufferCount {
    _pruneDeadReferences();
    return _activeBuffers.length;
  }

  /// Allocates a new [GpuBuffer] of [sizeInBytes] bytes on this device.
  ///
  /// It is an error if this device is disposed or if [sizeInBytes] is negative.
  GpuBuffer createBuffer({
    required int sizeInBytes,
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
  }) {
    _checkNotDisposed();
    return GpuBuffer.allocate(
      sizeInBytes: sizeInBytes,
      usage: usage,
      device: this,
    );
  }

  /// Allocates a new [GpuBuffer] of [sizeInBytes] bytes on this device and
  /// initializes it with [sizeInBytes] bytes copied from [hostPointer].
  ///
  /// It is an error if this device is disposed or if [sizeInBytes] is negative.
  GpuBuffer createBufferWithData(
    ffi.Pointer<ffi.Void> hostPointer,
    int sizeInBytes, [
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
  ]) {
    _checkNotDisposed();
    final buffer = GpuBuffer.allocate(
      sizeInBytes: sizeInBytes,
      usage: usage | GpuBufferUsage.copyDst,
      device: this,
    );
    if (sizeInBytes > 0) {
      buffer.copyFromHost(hostPointer, sizeInBytes);
    }
    return buffer;
  }

  /// Reads [sizeInBytes] bytes from [buffer] into [hostPointer].
  ///
  /// It is an error if this device is disposed.
  void readBufferIntoPointer(
    GpuBuffer buffer,
    ffi.Pointer<ffi.Void> hostPointer,
    int sizeInBytes,
  ) {
    _checkNotDisposed();
    buffer.copyToHost(hostPointer, sizeInBytes);
  }

  /// Registers an active [buffer] with this device for memory tracking.
  @internal
  void registerBuffer(GpuBuffer buffer) {
    _pruneDeadReferences();
    for (final ref in _activeBuffers) {
      if (identical(ref.target, buffer)) {
        return;
      }
    }
    _activeBuffers.add(WeakReference(buffer));
  }

  /// Unregisters a disposed [buffer] from this device's active tracking list.
  @internal
  void unregisterBuffer(GpuBuffer buffer) {
    for (var i = _activeBuffers.length - 1; i >= 0; i--) {
      final target = _activeBuffers[i].target;
      if (target == null || identical(target, buffer)) {
        _activeBuffers.removeAt(i);
      }
    }
  }

  void _pruneDeadReferences() {
    _activeBuffers.removeWhere(
      (ref) => ref.target == null || ref.target!.isDisposed,
    );
  }

  /// Synchronizes all queued operations on this device.
  ///
  /// It is an error if this device has been disposed.
  Future<void> synchronize() async {
    _checkNotDisposed();
  }

  /// Disposes all active buffers, purges the memory pool, and shuts down the backend.
  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    ResourceScope.untrack(this);

    final liveBuffers = <GpuBuffer>[];
    for (final ref in _activeBuffers) {
      final buffer = ref.target;
      if (buffer != null && !buffer.isDisposed) {
        liveBuffers.add(buffer);
      }
    }
    _activeBuffers.clear();

    for (final buffer in liveBuffers) {
      buffer.forceDisposeOnDeviceShutdown();
    }

    _memoryPool.dispose();
    _jitCompiler.clearCache();
    backend.dispose();
  }

  @override
  GpuDevice detachFromScope() {
    _checkNotDisposed();
    ResourceScope.untrack(this);
    return this;
  }

  @override
  GpuDevice detachToParentScope() {
    _checkNotDisposed();
    ResourceScope.promoteToParent(this);
    return this;
  }

  @override
  String toString() => 'GpuDevice(name: "$name", type: ${type.label})';
}
