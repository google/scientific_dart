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

/// An efficient power-of-2 size-bucketed memory pool allocator for GPU buffers.
///
/// Reduces native memory allocation overhead by caching and recycling freed
/// buffers of compatible sizes in O(1) time.
class GpuMemoryPool {
  final GpuDevice device;
  final int maxCachedBytes;
  final Map<int, List<GpuBuffer>> _buckets = {};

  int _cachedBytes = 0;
  int _hits = 0;
  int _misses = 0;
  bool _isDisposed = false;

  GpuMemoryPool(this.device, {this.maxCachedBytes = 256 * 1024 * 1024});

  /// Whether the memory pool has been disposed.
  bool get isDisposed => _isDisposed;

  /// Total number of bytes currently cached in the pool.
  int get cachedBytes => _cachedBytes;

  /// Number of successful cache hits (recycled buffers).
  int get hits => _hits;

  /// Number of cache misses (new native allocations).
  int get misses => _misses;

  /// Calculates the power-of-2 bucket size for a requested [sizeInBytes].
  static int computeBucketSize(int sizeInBytes) {
    if (sizeInBytes <= 64) return 64;
    var power = 64;
    while (power < sizeInBytes) {
      power <<= 1;
    }
    return power;
  }

  /// Acquires a buffer of at least [sizeInBytes] from the pool or allocates a new one.
  GpuBuffer acquire(
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.storage,
  }) {
    if (_isDisposed || device.isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot acquire buffer on disposed device or memory pool.',
      );
    }
    if (sizeInBytes < 0) {
      throw ArgumentError.value(
        sizeInBytes,
        'sizeInBytes',
        'Buffer size cannot be negative.',
      );
    }
    if (sizeInBytes == 0) {
      return GpuBuffer.unmanaged(ffi.nullptr, 0, device: device, usage: usage);
    }

    final bucketSize = computeBucketSize(sizeInBytes);
    final bucket = _buckets[bucketSize];

    if (bucket != null && bucket.isNotEmpty) {
      final buffer = bucket.removeLast();
      _cachedBytes -= bucketSize;
      _hits++;
      buffer.resetForReuse(usage: usage);
      return buffer;
    }

    _misses++;
    final ptr = device.backend.allocateBuffer(bucketSize);
    return GpuBuffer.fromPool(
      ptr,
      bucketSize,
      this,
      device: device,
      usage: usage,
    );
  }

  /// Recycles a buffer back into the pool.
  void recycle(GpuBuffer buffer) {
    if (buffer.rawPointer == ffi.nullptr) return;

    final bucketSize = buffer.sizeInBytes;
    if (_isDisposed || (_cachedBytes + bucketSize > maxCachedBytes)) {
      device.backend.freeBuffer(
        buffer.rawPointer.cast<ffi.Uint8>(),
        bucketSize,
      );
      return;
    }

    final bucket = _buckets.putIfAbsent(bucketSize, () => []);
    bucket.add(buffer);
    _cachedBytes += bucketSize;
  }

  /// Trims all cached buffers and releases native memory back to the OS.
  void trim() {
    for (final entry in _buckets.entries) {
      final bucket = entry.value;
      for (final buffer in bucket) {
        device.backend.freeBuffer(
          buffer.rawPointer.cast<ffi.Uint8>(),
          buffer.sizeInBytes,
        );
      }
      bucket.clear();
    }
    _cachedBytes = 0;
  }

  /// Disposes the pool and frees all cached buffers.
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    trim();
    _buckets.clear();
  }
}
