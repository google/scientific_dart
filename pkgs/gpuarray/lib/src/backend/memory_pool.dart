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

import '../buffer.dart';
import '../device.dart';
import '../exceptions.dart';

/// Power-of-two bucketed VRAM memory pool for O(1) buffer recycling on a [GpuDevice].
final class GpuMemoryPool {
  /// Minimum allocation bucket size in bytes (64 bytes).
  static const int minBucketSize = 64;

  /// The [GpuDevice] owning this memory pool.
  final GpuDevice device;

  /// Maximum total bytes that may be cached in free buckets before evicting.
  final int maxCachedBytes;

  final Map<int, List<GpuBuffer>> _freeBuckets = {};
  int _cachedBytes = 0;
  int _hits = 0;
  int _misses = 0;
  bool _isDisposed = false;

  /// Creates a [GpuMemoryPool] for [device] with an optional [maxCachedBytes] limit.
  GpuMemoryPool(this.device, {this.maxCachedBytes = 256 * 1024 * 1024});

  /// Rounds [sizeInBytes] up to the smallest power-of-two bucket size (`>= 64`).
  static int computeBucketSize(int sizeInBytes) {
    if (sizeInBytes <= minBucketSize) return minBucketSize;
    var bucket = minBucketSize;
    while (bucket < sizeInBytes) {
      bucket <<= 1;
    }
    return bucket;
  }

  void _checkNotDisposed() {
    if (_isDisposed || device.isDisposed) {
      throw GpuDeviceDisposedException(device.name);
    }
  }

  /// Whether this memory pool has been disposed.
  bool get isDisposed => _isDisposed;

  /// Total bytes currently held in free pool buckets awaiting reuse.
  int get cachedBytes => _cachedBytes;

  /// Number of buffer acquisitions satisfied from the free pool.
  int get hits => _hits;

  /// Number of buffer acquisitions that required a new backend allocation.
  int get misses => _misses;

  /// Acquires a [GpuBuffer] of at least [sizeInBytes] bytes from the pool,
  /// allocating a new bucket-aligned block if no free block is available.
  ///
  /// It is an error if [sizeInBytes] is negative or if [device] is disposed.
  GpuBuffer acquire(
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
  }) {
    _checkNotDisposed();
    RangeError.checkNotNegative(sizeInBytes, 'sizeInBytes');

    if (sizeInBytes == 0) {
      return GpuBuffer.pooled(
        device: device,
        nativeHandle: ffi.nullptr,
        sizeInBytes: 0,
        allocatedBytes: 0,
        usage: usage,
        owningPool: this,
      );
    }

    final bucketSize = computeBucketSize(sizeInBytes);
    if (_freeBuckets[bucketSize] case final bucket? when bucket.isNotEmpty) {
      final recycled = bucket.removeLast();
      _cachedBytes -= bucketSize;
      _hits++;
      recycled.reviveFromPool(requestedSize: sizeInBytes, newUsage: usage);
      if (recycled.rawNativeHandle != ffi.nullptr) {
        device.backend.clearBuffer(recycled, offset: 0, bytes: bucketSize);
      }
      return recycled;
    }

    _misses++;
    final handle = device.backend.allocateBuffer(bucketSize, usage: usage);
    return GpuBuffer.pooled(
      device: device,
      nativeHandle: handle,
      sizeInBytes: sizeInBytes,
      allocatedBytes: bucketSize,
      usage: usage,
      owningPool: this,
    );
  }

  /// Releases a disposed pooled [buffer] back to its power-of-two bucket, or
  /// frees it immediately if the pool is disposed or full.
  @internal
  void release(GpuBuffer buffer) {
    final bucketSize = buffer.allocatedBytes;
    if (_isDisposed ||
        device.isDisposed ||
        bucketSize <= 0 ||
        _cachedBytes + bucketSize > maxCachedBytes) {
      if (buffer.rawNativeHandle != ffi.nullptr) {
        device.backend.freeBuffer(buffer.rawNativeHandle, bucketSize);
      }
      return;
    }

    final bucket = _freeBuckets[bucketSize] ??= <GpuBuffer>[];
    bucket.add(buffer);
    _cachedBytes += bucketSize;
  }

  /// Alias for [release] for recycling a buffer back into the pool.
  @internal
  void recycle(GpuBuffer buffer) => release(buffer);

  /// Frees all currently cached idle buffers in this pool back to the OS/driver.
  void trim() {
    for (final bucket in _freeBuckets.values) {
      for (final buffer in bucket) {
        if (buffer.rawNativeHandle != ffi.nullptr) {
          device.backend.freeBuffer(
            buffer.rawNativeHandle,
            buffer.allocatedBytes,
          );
        }
      }
      bucket.clear();
    }
    _freeBuckets.clear();
    _cachedBytes = 0;
  }

  /// Purges all cached buffers and marks this pool disposed.
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    trim();
  }
}
