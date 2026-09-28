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
import 'package:ffi/ffi.dart' as pkg_ffi;

import 'pocketfft_bindings.dart';

/// Supported types of PocketFFT / KissFFT transformation plans.
enum PocketFFTPlanType {
  /// 1D Complex FFT plan ([kiss_fft_cfg]).
  complex1D,

  /// 1D Real-optimized FFT plan ([kiss_fftr_cfg]).
  real1D,

  /// N-Dimensional Complex FFT plan ([kiss_fftnd_cfg]).
  complexND,
}

/// Base key representing a unique PocketFFT plan configuration.
sealed class PocketFFTPlanKey {
  const PocketFFTPlanKey();
}

/// Cache key for a 1D complex FFT plan ([kiss_fft_cfg]).
final class Complex1DPlanKey extends PocketFFTPlanKey {
  /// The transform length ($N$).
  final int length;

  /// Whether this plan computes an inverse transform ($1$) or forward ($0$).
  final bool isInverse;

  /// Creates a key for a 1D complex FFT plan.
  const Complex1DPlanKey(this.length, {this.isInverse = false});

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Complex1DPlanKey &&
          runtimeType == other.runtimeType &&
          length == other.length &&
          isInverse == other.isInverse;

  @override
  int get hashCode =>
      Object.hash(length, isInverse, PocketFFTPlanType.complex1D);

  @override
  String toString() =>
      'Complex1DPlanKey(length: $length, isInverse: $isInverse)';
}

/// Cache key for a 1D real FFT plan ([kiss_fftr_cfg]).
final class Real1DPlanKey extends PocketFFTPlanKey {
  /// The real signal transform length ($N$).
  final int length;

  /// Whether this plan computes an inverse transform ($1$) or forward ($0$).
  final bool isInverse;

  /// Creates a key for a 1D real FFT plan.
  const Real1DPlanKey(this.length, {this.isInverse = false});

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Real1DPlanKey &&
          runtimeType == other.runtimeType &&
          length == other.length &&
          isInverse == other.isInverse;

  @override
  int get hashCode => Object.hash(length, isInverse, PocketFFTPlanType.real1D);

  @override
  String toString() => 'Real1DPlanKey(length: $length, isInverse: $isInverse)';
}

/// Cache key for an N-dimensional complex FFT plan ([kiss_fftnd_cfg]).
final class ComplexNDPlanKey extends PocketFFTPlanKey {
  /// The dimensions of the transform.
  final List<int> dimensions;

  /// Whether this plan computes an inverse transform ($1$) or forward ($0$).
  final bool isInverse;

  /// Creates a key for an N-dimensional complex FFT plan.
  ComplexNDPlanKey(List<int> dimensions, {this.isInverse = false})
    : dimensions = List.unmodifiable(dimensions);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! ComplexNDPlanKey ||
        runtimeType != other.runtimeType ||
        isInverse != other.isInverse ||
        dimensions.length != other.dimensions.length) {
      return false;
    }
    for (var i = 0; i < dimensions.length; i++) {
      if (dimensions[i] != other.dimensions[i]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hash(
    Object.hashAll(dimensions),
    isInverse,
    PocketFFTPlanType.complexND,
  );

  @override
  String toString() =>
      'ComplexNDPlanKey(dimensions: $dimensions, isInverse: $isInverse)';
}

/// An isolate-level cache for PocketFFT / KissFFT plans.
///
/// Reuses precomputed twiddle factors and factorization configurations
/// across transforms of the same dimensions and direction, eliminating
/// the per-call allocation and trigonometric initialization overhead.
///
/// Employs a Least-Recently-Used (LRU) eviction strategy when the number of
/// cached plans exceeds [maxCapacity].
///
/// **Thread Safety:**
/// In Dart, each isolate maintains its own independent memory heap and
/// static variables. Therefore, [PocketFFTPlanCache.instance] is strictly
/// isolate-local and thread-safe without locks or mutexes.
///
/// **Memory Ownership & Lifetime:**
/// Plans are allocated on the unmanaged C heap using native `malloc`. When
/// evicted by LRU or cleared via [clear] / [dispose], memory is freed using C `free`.
final class PocketFFTPlanCache {
  /// Default maximum number of plans retained in the cache.
  static const int defaultMaxCapacity = 256;

  /// The default isolate-local singleton instance.
  static final PocketFFTPlanCache instance = PocketFFTPlanCache();

  int _maxCapacity;
  final Map<PocketFFTPlanKey, ffi.Pointer<ffi.Void>> _cache = {};
  bool _isDisposed = false;

  /// Creates a new [PocketFFTPlanCache] with the specified [maxCapacity].
  ///
  /// It is an error if [maxCapacity] is less than or equal to 0.
  PocketFFTPlanCache({int maxCapacity = defaultMaxCapacity})
    : _maxCapacity = maxCapacity {
    if (maxCapacity <= 0) {
      throw ArgumentError.value(
        maxCapacity,
        'maxCapacity',
        'Cache capacity must be strictly positive',
      );
    }
  }

  /// Maximum number of plans to store in this cache before LRU eviction.
  int get maxCapacity => _maxCapacity;

  /// Updates the maximum capacity of this cache.
  ///
  /// If the new capacity is less than the current cache size, the least
  /// recently used plans will be evicted and freed immediately.
  ///
  /// It is an error if [value] is less than or equal to 0.
  /// It is an error if this cache is disposed.
  set maxCapacity(int value) {
    if (_isDisposed) {
      throw StateError('Cannot modify capacity of a disposed cache.');
    }
    if (value <= 0) {
      throw ArgumentError.value(
        value,
        'maxCapacity',
        'Cache capacity must be strictly positive',
      );
    }
    _maxCapacity = value;
    while (_cache.length > _maxCapacity) {
      final oldestKey = _cache.keys.first;
      final oldestPtr = _cache.remove(oldestKey)!;
      pkg_ffi.malloc.free(oldestPtr);
    }
  }

  /// Current number of plans stored in this cache.
  int get size => _cache.length;

  /// Current number of plans stored in this cache (alias for [size]).
  int get length => _cache.length;

  /// Returns `true` if this cache has been disposed.
  bool get isDisposed => _isDisposed;

  /// Retrieves or creates a 1D complex FFT configuration for the given [length]
  /// and direction [isInverse].
  ///
  /// Precomputed plans are cached and reused on subsequent calls with matching
  /// arguments.
  ///
  /// **Preconditions:**
  /// - It is an error if [length] is less than or equal to 0.
  /// - It is an error if this cache is disposed.
  ///
  /// **Throws:**
  /// - [StateError] if native memory allocation or plan configuration fails.
  ///
  /// **Performance considerations:**
  /// - Cache hit: $O(1)$ lookup time without heap allocations or trigonometric precalculations.
  /// - Cache miss: $O(N)$ initialization of twiddle factor array.
  kiss_fft_cfg getPlan(int length, {bool isInverse = false}) {
    if (_isDisposed) {
      throw StateError('Cannot retrieve plans from a disposed cache.');
    }
    if (length <= 0) {
      throw ArgumentError.value(
        length,
        'length',
        'Transform length must be strictly positive',
      );
    }

    final key = Complex1DPlanKey(length, isInverse: isInverse);
    final cached = _cache.remove(key);
    if (cached != null) {
      _cache[key] = cached;
      return cached.cast<kiss_fft_state>();
    }

    final pLen = pkg_ffi.malloc<ffi.Size>();
    try {
      pLen.value = 0;
      kiss_fft_alloc(length, isInverse ? 1 : 0, ffi.nullptr, pLen);
      final needed = pLen.value;
      if (needed <= 0) {
        throw StateError(
          'Failed to query native kiss_fft_cfg plan size for length $length',
        );
      }
      final buf = pkg_ffi.malloc<ffi.Uint8>(needed);
      final cfg = kiss_fft_alloc(
        length,
        isInverse ? 1 : 0,
        buf.cast<ffi.Void>(),
        pLen,
      );
      if (cfg.address == 0) {
        pkg_ffi.malloc.free(buf);
        throw StateError(
          'Failed to allocate native kiss_fft_cfg plan for length $length',
        );
      }

      _insert(key, cfg.cast<ffi.Void>());
      return cfg;
    } finally {
      pkg_ffi.malloc.free(pLen);
    }
  }

  /// Retrieves or creates a 1D real FFT configuration for the given [length]
  /// and direction [isInverse].
  ///
  /// Precomputed plans are cached and reused on subsequent calls with matching
  /// arguments.
  ///
  /// **Preconditions:**
  /// - It is an error if [length] is less than or equal to 0.
  /// - It is an error if this cache is disposed.
  ///
  /// **Throws:**
  /// - [StateError] if native memory allocation or plan configuration fails.
  ///
  /// **Performance considerations:**
  /// - Cache hit: $O(1)$ lookup time.
  /// - Cache miss: $O(N)$ initialization.
  kiss_fftr_cfg getRealPlan(int length, {bool isInverse = false}) {
    if (_isDisposed) {
      throw StateError('Cannot retrieve plans from a disposed cache.');
    }
    if (length <= 0) {
      throw ArgumentError.value(
        length,
        'length',
        'Transform length must be strictly positive',
      );
    }

    final key = Real1DPlanKey(length, isInverse: isInverse);
    final cached = _cache.remove(key);
    if (cached != null) {
      _cache[key] = cached;
      return cached.cast<kiss_fftr_state>();
    }

    final pLen = pkg_ffi.malloc<ffi.Size>();
    try {
      pLen.value = 0;
      kiss_fftr_alloc(length, isInverse ? 1 : 0, ffi.nullptr, pLen);
      final needed = pLen.value;
      if (needed <= 0) {
        throw StateError(
          'Failed to query native kiss_fftr_cfg plan size for length $length',
        );
      }
      final buf = pkg_ffi.malloc<ffi.Uint8>(needed);
      final cfg = kiss_fftr_alloc(
        length,
        isInverse ? 1 : 0,
        buf.cast<ffi.Void>(),
        pLen,
      );
      if (cfg.address == 0) {
        pkg_ffi.malloc.free(buf);
        throw StateError(
          'Failed to allocate native kiss_fftr_cfg plan for length $length',
        );
      }

      _insert(key, cfg.cast<ffi.Void>());
      return cfg;
    } finally {
      pkg_ffi.malloc.free(pLen);
    }
  }

  /// Retrieves or creates an N-dimensional complex FFT configuration for the given [dimensions]
  /// and direction [isInverse].
  ///
  /// Precomputed plans are cached and reused on subsequent calls with matching
  /// arguments.
  ///
  /// **Preconditions:**
  /// - It is an error if [dimensions] is empty or contains non-positive values.
  /// - It is an error if this cache is disposed.
  ///
  /// **Throws:**
  /// - [StateError] if native memory allocation or plan configuration fails.
  ///
  /// **Performance considerations:**
  /// - Cache hit: $O(1)$ lookup time.
  /// - Cache miss: $O(\prod dimensions)$ initialization and allocation of multi-stage buffers.
  kiss_fftnd_cfg getNDPlan(List<int> dimensions, {bool isInverse = false}) {
    if (_isDisposed) {
      throw StateError('Cannot retrieve plans from a disposed cache.');
    }
    if (dimensions.isEmpty) {
      throw ArgumentError.value(
        dimensions,
        'dimensions',
        'Dimensions list must not be empty',
      );
    }
    for (final d in dimensions) {
      if (d <= 0) {
        throw ArgumentError.value(
          dimensions,
          'dimensions',
          'All dimension sizes must be strictly positive',
        );
      }
    }

    final key = ComplexNDPlanKey(dimensions, isInverse: isInverse);
    final cached = _cache.remove(key);
    if (cached != null) {
      _cache[key] = cached;
      return cached.cast<kiss_fftnd_state>();
    }

    final ndims = dimensions.length;
    final pDims = pkg_ffi.malloc<ffi.Int>(ndims);
    final pLen = pkg_ffi.malloc<ffi.Size>();
    try {
      for (var i = 0; i < ndims; i++) {
        pDims[i] = dimensions[i];
      }
      pLen.value = 0;
      kiss_fftnd_alloc(pDims, ndims, isInverse ? 1 : 0, ffi.nullptr, pLen);
      final needed = pLen.value;
      if (needed <= 0) {
        throw StateError(
          'Failed to query native kiss_fftnd_cfg plan size for dimensions $dimensions',
        );
      }
      final buf = pkg_ffi.malloc<ffi.Uint8>(needed);
      final cfg = kiss_fftnd_alloc(
        pDims,
        ndims,
        isInverse ? 1 : 0,
        buf.cast<ffi.Void>(),
        pLen,
      );
      if (cfg.address == 0) {
        pkg_ffi.malloc.free(buf);
        throw StateError(
          'Failed to allocate native kiss_fftnd_cfg plan for dimensions $dimensions',
        );
      }
      _insert(key, cfg.cast<ffi.Void>());
      return cfg;
    } finally {
      pkg_ffi.malloc.free(pDims);
      pkg_ffi.malloc.free(pLen);
    }
  }

  void _insert(PocketFFTPlanKey key, ffi.Pointer<ffi.Void> ptr) {
    while (_cache.length >= _maxCapacity) {
      final oldestKey = _cache.keys.first;
      final oldestPtr = _cache.remove(oldestKey)!;
      pkg_ffi.malloc.free(oldestPtr);
    }
    _cache[key] = ptr;
  }

  /// Frees all cached native FFT plan configurations and clears the cache.
  ///
  /// It is an error if this cache is disposed.
  void clear() {
    if (_isDisposed) {
      throw StateError('Cannot clear a disposed cache.');
    }
    for (final ptr in _cache.values) {
      pkg_ffi.malloc.free(ptr);
    }
    _cache.clear();
  }

  /// Disposes this cache and frees all associated native plan memory.
  void dispose() {
    if (_isDisposed) return;
    for (final ptr in _cache.values) {
      pkg_ffi.malloc.free(ptr);
    }
    _cache.clear();
    _isDisposed = true;
  }
}

/// Convenience helper to get a cached 1D complex FFT plan from the default isolate cache.
kiss_fft_cfg getCachedKissFFTPlan(int length, {bool isInverse = false}) =>
    PocketFFTPlanCache.instance.getPlan(length, isInverse: isInverse);

/// Convenience helper to get a cached 1D real FFT plan from the default isolate cache.
kiss_fftr_cfg getCachedKissFFTRPlan(int length, {bool isInverse = false}) =>
    PocketFFTPlanCache.instance.getRealPlan(length, isInverse: isInverse);

/// Convenience helper to get a cached N-dimensional complex FFT plan from the default isolate cache.
kiss_fftnd_cfg getCachedKissFFTNDPlan(
  List<int> dimensions, {
  bool isInverse = false,
}) => PocketFFTPlanCache.instance.getNDPlan(dimensions, isInverse: isInverse);

/// Convenience helper to clear all plans from the default isolate cache.
void clearPocketFFTPlanCache() => PocketFFTPlanCache.instance.clear();
