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
import 'dart:math' show min;
import 'package:ffi/ffi.dart';
import 'ndarray.dart' show Complex, ComplexList;
import 'wasm_pointer_lists.dart' show isWasmRuntime, maxWasm32AllocationBytes;

/// An Isolate-local scratch memory arena for transient FFI allocations.
///
/// Maintains pre-allocated persistent C heap memory to reduce allocation overhead.
final class ScratchArena {
  ScratchArena._();

  static final List<ffi.Pointer<ffi.Uint8>> _pages = [];
  static final List<int> _pageCapacities = [];
  static const int _baseCapacity = 256 * 1024; // 256KB base capacity for page 0
  static int _currentPageIndex = 0;
  static int _offset = 0;

  static ffi.Pointer<ffi.Uint8> _mallocPage(int bytes) {
    try {
      return malloc<ffi.Uint8>(bytes);
      // package:ffi's malloc throws ArgumentError when OS allocation fails.
      // ignore: avoid_catching_errors
    } on ArgumentError {
      throw OutOfMemoryError();
    }
  }

  static void _init() {
    if (_pages.isEmpty) {
      final page = _mallocPage(_baseCapacity);
      _pages.add(page);
      _pageCapacities.add(_baseCapacity);
    }
  }

  /// Allocates [bytes] of memory from the arena stack, aligned to 8 bytes.
  ///
  /// **Preconditions:**
  /// - [bytes] must be non-negative.
  ///
  /// It is an error if [bytes] is negative, or if the native allocator cannot
  /// satisfy the allocation request (throws [OutOfMemoryError]).
  ///
  /// **Performance considerations:**
  /// - Amortized $O(1)$ complexity. If the current page has enough space,
  ///   allocation is a simple pointer offset increment.
  /// - If a new page needs to be allocated, it invokes native `malloc`, which
  ///   has $O(1)$ to $O(N)$ complexity depending on the system allocator.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<T> allocate<T extends ffi.NativeType>(int bytes) {
    if (bytes < 0) {
      throw ArgumentError.value(bytes, 'bytes', 'Must be non-negative.');
    }
    // Both bounds leave 16 bytes of headroom for the alignment rounding below;
    // the wasm32 bound only applies where `size_t` is 32 bits wide.
    if (bytes > 0x7ffffffffffffff0 ||
        (ffi.sizeOf<ffi.Size>() == 4 &&
            bytes > (maxWasm32AllocationBytes & ~0xF))) {
      throw OutOfMemoryError();
    }
    _init();

    // Ensure 8-byte alignment for native FFI alignment requirements
    final alignedBytes = (bytes + 7) & ~7;

    final currentCapacity = _pageCapacities[_currentPageIndex];

    // If current page doesn't have enough space, switch to next page!
    // Allocate any required new page BEFORE mutating _currentPageIndex,
    // _offset, _pages, or _pageCapacities so failed allocations leave the
    // arena in a valid, uncorrupted state.
    if (alignedBytes > currentCapacity - _offset) {
      final nextPageIndex = _currentPageIndex + 1;

      if (nextPageIndex >= _pages.length) {
        // Geometrically scale capacities: Page 0 (256KB), Page 1 (512KB), Page 2 (1MB) etc.
        final scaledCap = min(1 << 30, _baseCapacity << min(nextPageIndex, 20));
        final targetCap = alignedBytes > scaledCap ? alignedBytes : scaledCap;
        final newPage = _mallocPage(targetCap);
        _pages.add(newPage);
        _pageCapacities.add(targetCap);
      } else if (_pageCapacities[nextPageIndex] < alignedBytes) {
        // If reusing a cached page but its capacity is too small for this allocation,
        // allocate the custom-sized page first before mutating arena state.
        final newPage = _mallocPage(alignedBytes);

        // Instead of freeing the small page, move it to the end of the list
        // so it can be reused later for standard page requests.
        final smallPage = _pages.removeAt(nextPageIndex);
        final smallCap = _pageCapacities.removeAt(nextPageIndex);
        _pages.add(smallPage);
        _pageCapacities.add(smallCap);

        // Insert the new custom-sized page at nextPageIndex.
        _pages.insert(nextPageIndex, newPage);
        _pageCapacities.insert(nextPageIndex, alignedBytes);
      }

      _currentPageIndex = nextPageIndex;
      _offset = 0;
    }

    final currentArena = _pages[_currentPageIndex];
    final ptr = ffi.Pointer<T>.fromAddress(currentArena.address + _offset);
    _offset += alignedBytes;
    return ptr;
  }

  /// Gets the current stack marker in the arena.
  static ScratchMarker get marker =>
      ScratchMarker._(_currentPageIndex, _offset);

  /// Resets the arena stack back to the given [marker].
  ///
  /// This rolls back the allocation offset and deallocates any excess pages.
  /// Empty custom-allocated pages are pruned even if they are below the
  /// standard preservation limit (page 0 or 1).
  ///
  /// **Preconditions:**
  /// - [marker] must be a valid marker obtained from [ScratchArena.marker]
  ///   in the current allocation session.
  /// - Cannot reset to a marker that is ahead of the current stack pointer.
  ///
  /// **Performance considerations:**
  /// - $O(P)$ where $P$ is the number of pages pruned. Usually $O(1)$ unless
  ///   many pages were allocated.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static void reset(ScratchMarker marker) {
    final pageIndex = marker.pageIndex;
    final offset = marker.offset;

    if (pageIndex < 0 ||
        pageIndex > _currentPageIndex ||
        offset < 0 ||
        (pageIndex == _currentPageIndex && offset > _offset)) {
      throw StateError(
        'Invalid or stale ScratchMarker: cannot reset ahead of current stack pointer or to an out-of-order marker.',
      );
    }

    int? pruneFromIndex;
    for (var i = 0; i < _pages.length; i++) {
      final isEmpty = i > pageIndex || (i == pageIndex && offset == 0);
      if (isEmpty) {
        final isCustom = _pageCapacities[i] > (_baseCapacity << min(i, 20));
        final isExcess = i >= 2;
        if (isCustom || isExcess) {
          pruneFromIndex = i;
          break;
        }
      }
    }

    if (pruneFromIndex != null) {
      while (_pages.length > pruneFromIndex) {
        final page = _pages.removeLast();
        _pageCapacities.removeLast();
        malloc.free(page);
      }
    }

    if (_pages.isEmpty) {
      _currentPageIndex = 0;
      _offset = 0;
    } else if (pageIndex >= _pages.length) {
      _currentPageIndex = _pages.length - 1;
      _offset = _pageCapacities[_currentPageIndex];
    } else {
      _currentPageIndex = pageIndex;
      _offset = offset;
    }
  }

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Int64]s.
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Transient allocation in the pre-allocated C stack.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Int64> copyInts(List<int> list) => copyInt64s(list);

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Double]s.
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Uses fast typed list views to copy contiguous memory blocks.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Double> copyDoubles(List<double> list) {
    final ptr = allocate<ffi.Double>(list.length * ffi.sizeOf<ffi.Double>());
    if (isWasmRuntime) {
      for (var i = 0; i < list.length; i++) {
        ptr[i] = list[i];
      }
    } else {
      ptr.asTypedList(list.length).setRange(0, list.length, list);
    }
    return ptr;
  }

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Float]s.
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Uses fast typed list views to copy contiguous memory blocks.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Float> copyFloats(List<double> list) {
    final ptr = allocate<ffi.Float>(list.length * ffi.sizeOf<ffi.Float>());
    if (isWasmRuntime) {
      for (var i = 0; i < list.length; i++) {
        ptr[i] = list[i];
      }
    } else {
      ptr.asTypedList(list.length).setRange(0, list.length, list);
    }
    return ptr;
  }

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Int32]s.
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Uses fast typed list views to copy contiguous memory blocks.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Int32> copyInt32s(List<int> list) {
    final ptr = allocate<ffi.Int32>(list.length * ffi.sizeOf<ffi.Int32>());
    for (var i = 0; i < list.length; i++) {
      final v = list[i];
      if (v < -0x80000000 || v > 0x7fffffff) {
        throw UnsupportedError('Value $v exceeds 32-bit native int limit.');
      }
      ptr[i] = v;
    }
    return ptr;
  }

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Int64]s.
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Uses fast typed list views to copy contiguous memory blocks.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Int64> copyInt64s(List<int> list) {
    final ptr = allocate<ffi.Int64>(list.length * ffi.sizeOf<ffi.Int64>());
    if (isWasmRuntime) {
      for (var i = 0; i < list.length; i++) {
        ptr[i] = list[i];
      }
    } else {
      ptr.asTypedList(list.length).setRange(0, list.length, list);
    }
    return ptr;
  }

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Double]s.
  ///
  /// Each complex number is represented as 2 consecutive double values (real followed by imaginary).
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Specially optimized for [ComplexList] to perform a direct contiguous memory copy.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Double> copyComplexes(List<Complex> list) {
    final ptr = allocate<ffi.Double>(
      list.length * 2 * ffi.sizeOf<ffi.Double>(),
    );
    if (list is ComplexList) {
      final backing = list.backingList;
      final total = list.length * 2;
      if (isWasmRuntime) {
        for (var i = 0; i < total; i++) {
          ptr[i] = backing[i];
        }
      } else {
        ptr.asTypedList(total).setRange(0, total, backing);
      }
    } else {
      for (var i = 0; i < list.length; i++) {
        ptr[i * 2] = list[i].real;
        ptr[i * 2 + 1] = list[i].imag;
      }
    }
    return ptr;
  }

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Float]s.
  ///
  /// Each complex number is represented as 2 consecutive float values (real followed by imaginary).
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Specially optimized for [ComplexList] to perform a direct contiguous memory copy.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Float> copyFloatComplexes(List<Complex> list) {
    final ptr = allocate<ffi.Float>(list.length * 2 * ffi.sizeOf<ffi.Float>());
    if (list is ComplexList) {
      final backing = list.backingList;
      final total = list.length * 2;
      if (isWasmRuntime) {
        for (var i = 0; i < total; i++) {
          ptr[i] = backing[i];
        }
      } else {
        ptr.asTypedList(total).setRange(0, total, backing);
      }
    } else {
      for (var i = 0; i < list.length; i++) {
        ptr[i * 2] = list[i].real;
        ptr[i * 2 + 1] = list[i].imag;
      }
    }
    return ptr;
  }

  /// Allocates transient memory from the arena and copies the elements of [list] into it as native [ffi.Uint8] bytes (1 for true, 0 for false).
  ///
  /// **Preconditions:**
  /// - [list] must be non-null.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(N)$ where $N$ is the number of elements in [list].
  /// - Fast element-wise iteration to map boolean states to native byte flags.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Uint8> copyBools(List<bool> list) {
    final ptr = allocate<ffi.Uint8>(list.length * ffi.sizeOf<ffi.Uint8>());
    for (var i = 0; i < list.length; i++) {
      ptr[i] = list[i] ? 1 : 0;
    }
    return ptr;
  }

  /// Allocates transient memory from the arena for strided operations.
  ///
  /// Allocations are bump-allocated on the [ScratchArena] stack, making them
  /// safe for reentrant and nested operations when callers use [marker] and [reset].
  ///
  /// [ndim] is the number of dimensions.
  /// [segments] is the number of segments of size [ndim] needed in the buffer.
  /// Defaults to 4.
  ///
  /// **Preconditions:**
  /// - [ndim] must be non-negative.
  /// - [segments] must be non-negative.
  ///
  /// It is an error if [ndim] or [segments] is negative.
  ///
  /// **Performance considerations:**
  /// - Time complexity is $O(1)$ amortized allocation on the bump stack.
  ///
  /// **Example:**
  /// {@example /example/scratch_arena_example.dart lang=dart}
  static ffi.Pointer<ffi.Int64> getStridedBuffer(int ndim, [int segments = 4]) {
    if (ndim < 0) {
      throw ArgumentError.value(ndim, 'ndim', 'Must be non-negative.');
    }
    if (segments < 0) {
      throw ArgumentError.value(segments, 'segments', 'Must be non-negative.');
    }
    if (segments < 4) segments = 4;
    final count = ndim * segments;
    final requiredSize = count > 0 ? count : 1;
    return allocate<ffi.Int64>(requiredSize * ffi.sizeOf<ffi.Int64>());
  }

  /// Releases all persistent resources held by the [ScratchArena].
  ///
  /// This frees all pre-allocated pages.
  /// Clients should call this when they are done using the arena to free
  /// native memory.
  ///
  /// It is an error to call [cleanup] while arena allocations are active
  /// (i.e. before all markers have been reset back to the root offset).
  static void cleanup() {
    if (_currentPageIndex != 0 || _offset != 0) {
      throw StateError(
        'Cannot clean up ScratchArena while allocations are active; '
        'reset all markers before calling cleanup().',
      );
    }
    for (final page in _pages) {
      malloc.free(page);
    }
    _pages.clear();
    _pageCapacities.clear();
    _currentPageIndex = 0;
    _offset = 0;
  }
}

/// Represents a stable checkpoint marker for the [ScratchArena] memory stack.
///
/// Markers are used to reset the arena stack back to a previous state,
/// effectively freeing all allocations made after the marker was recorded.
///
/// **Example:**
/// {@example /example/scratch_arena_example.dart lang=dart}
final class ScratchMarker {
  /// The page index inside the ScratchArena page pool when the marker was recorded.
  final int pageIndex;

  /// The byte offset inside the page when the marker was recorded.
  final int offset;

  const ScratchMarker._(this.pageIndex, this.offset);

  @override
  bool operator ==(Object other) =>
      other is ScratchMarker &&
      other.pageIndex == pageIndex &&
      other.offset == offset;

  @override
  int get hashCode => Object.hash(pageIndex, offset);

  @override
  String toString() => 'ScratchMarker(pageIndex: $pageIndex, offset: $offset)';
}
