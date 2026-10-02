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
import 'dart:isolate';
import 'dart:typed_data';

import 'ndarray.dart';
import 'wasm_pointer_lists.dart' show isWasmRuntime;

/// Transmission mode for a [SendableNDArray].
enum _SendableMode { copy, borrow }

/// An isolate-sendable wrapper for [NDArray] enabling cross-isolate data transfer.
///
/// Ordinary [NDArray] instances cannot be directly passed across Dart Isolates
/// (such as via `Isolate.run` or [SendPort]) because:
/// 1. [NDArray] registers an [ffi.NativeFinalizer] token with the isolate-local
///    garbage collector. Passing an object bound to a native finalizer across
///    isolates is forbidden and unsafe.
/// 2. [NDArray] holds references to isolate-specific [ResourceScope] zones and
///    hierarchical view parent chains (`_parent`), which cannot be accessed
///    safely across isolate boundaries.
/// 3. Attaching independent native finalizers to the same unmanaged C pointer
///    across multiple isolates causes race conditions and catastrophic double-free crashes.
///
/// [SendableNDArray] resolves this by providing two explicit transmission modes:
/// - **Copy Mode** ([SendableNDArray.fromCopy] or [NDArray.toSendable]):
///   Copies the array data into a [TransferableTypedData] buffer alongside metadata
///   ([shape] and [dtypeIndex]). When received by another isolate, calling [materialize]
///   reconstructs a fresh, independent, scope-registered [NDArray] that owns its newly
///   allocated native C memory.
/// - **Borrow Mode** ([SendableNDArray.unsafeBorrow] or [NDArray.toSendableBorrow]):
///   Captures the raw native memory address ([address]), [shape], [strides], [dtypeIndex],
///   and [_physicalByteCapacity] without copying. The receiving isolate calls [materializeView]
///   to construct a zero-copy, non-owning view backed by [NDArray.fromPointer] (with no finalizer).
///   The caller is responsible for ensuring that the source array remains alive in its scope
///   on the sending isolate for the duration of the isolate task (for example, by awaiting
///   `Isolate.run` inside an [NDArray.scope]).
///
/// {@example /example/sendable_ndarray_example.dart}
final class SendableNDArray<T extends DTypeTag> {
  static const bool _isWasm = isWasmRuntime;

  final _SendableMode _mode;
  final TransferableTypedData? _transferableData;
  final Uint8List? _wasmBytes;
  final int? _address;
  final int? _physicalByteCapacity;

  /// The dimensions of the array.
  final List<int> shape;

  /// The number of elements to skip in memory along each dimension, if strided or borrowed.
  final List<int>? strides;

  /// The integer index of the array's [DType] in [DType.values].
  final int dtypeIndex;

  bool _isMaterialized = false;

  SendableNDArray._({
    required _SendableMode mode,
    required this.shape,
    required this.dtypeIndex,
    TransferableTypedData? transferableData,
    Uint8List? wasmBytes,
    int? address,
    this.strides,
    int? physicalByteCapacity,
  }) : _mode = mode,
       _transferableData = transferableData,
       _wasmBytes = wasmBytes,
       _address = address,
       _physicalByteCapacity = physicalByteCapacity;

  /// Copies [totalBytes] bytes starting at [src] into a fresh [Uint8List] one
  /// byte at a time.
  ///
  /// Only used on `dart2wasm`, where `Pointer.asTypedList` is unavailable; the
  /// VM path copies the `asTypedList` view in bulk instead.
  static Uint8List _copyPointerBytes(
    ffi.Pointer<ffi.Uint8> src,
    int totalBytes,
  ) {
    final bytes = Uint8List(totalBytes);
    for (var i = 0; i < totalBytes; i++) {
      bytes[i] = src[i];
    }
    return bytes;
  }

  /// Creates a [SendableNDArray] by copying [array]'s memory into a [TransferableTypedData].
  ///
  /// The resulting [SendableNDArray] is self-contained and completely independent of [array]'s
  /// lifetime. The receiver can call [materialize] to reconstruct a new [NDArray] owning its
  /// native memory.
  ///
  /// **Preconditions:**
  /// - [array] must not be disposed.
  ///
  /// It is an error if [array] has been disposed.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(N)$ where $N$ is the number of elements in [array].
  /// - Space complexity: $O(N)$ to allocate the transferable buffer.
  ///
  /// **Example:**
  /// ```dart
  /// final array = NDArray<Float64>.ones([10, 10], DType.float64);
  /// final sendable = SendableNDArray.fromCopy(array);
  /// final sum = await Isolate.run(() {
  ///   final workerArray = sendable.materialize();
  ///   return workerArray.sum().scalar;
  /// });
  /// ```
  factory SendableNDArray.fromCopy(NDArray<T> array) {
    if (array.isDisposed) {
      throw StateError('Cannot create SendableNDArray from a disposed array.');
    }
    final totalBytes = array.size * array.dtype.byteWidth;
    final TransferableTypedData? transferable;
    final Uint8List? wasmBytes;
    if (totalBytes == 0) {
      transferable = _isWasm
          ? null
          : TransferableTypedData.fromList([Uint8List(0)]);
      wasmBytes = _isWasm ? Uint8List(0) : null;
    } else {
      final contiguous = array.isContiguous ? array : array.copy();
      try {
        final src = contiguous.pointer.cast<ffi.Uint8>();
        if (_isWasm) {
          transferable = null;
          wasmBytes = _copyPointerBytes(src, totalBytes);
        } else {
          // `TransferableTypedData.fromList` copies the view, so this is the
          // only copy on the VM.
          final bytes = src.asTypedList(totalBytes);
          transferable = TransferableTypedData.fromList([bytes]);
          wasmBytes = null;
        }
      } finally {
        if (!identical(contiguous, array)) {
          contiguous.dispose();
        }
      }
    }
    return SendableNDArray._(
      mode: _SendableMode.copy,
      transferableData: transferable,
      wasmBytes: wasmBytes,
      shape: List<int>.unmodifiable(array.shape),
      dtypeIndex: array.dtype.index,
    );
  }

  /// Creates a zero-copy [SendableNDArray] borrowing the raw native memory address of [array].
  ///
  /// **Safety Contract:**
  /// - The caller MUST ensure that [array] remains allocated and alive on the source isolate
  ///   for the entire duration that the worker isolate accesses it. Typically, this is done by
  ///   wrapping execution in [NDArray.scope] on the main isolate and awaiting `Isolate.run`.
  /// - The worker isolate accesses memory directly in-place via [materializeView].
  /// - Concurrent unsynchronized writes to the same memory addresses result in undefined behavior.
  ///
  /// **Preconditions:**
  /// - [array] must not be disposed.
  ///
  /// It is an error if [array] has been disposed.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(1)$.
  /// - Space complexity: $O(1)$.
  ///
  /// **Example:**
  /// ```dart
  /// final array = NDArray<Float64>.zeros([100], DType.float64);
  /// final sendable = SendableNDArray.unsafeBorrow(array);
  /// await Isolate.run(() {
  ///   final view = sendable.materializeView();
  ///   view.fill(1.0);
  /// });
  /// print(array[0]); // 1.0
  /// ```
  factory SendableNDArray.unsafeBorrow(NDArray<T> array) {
    if (array.isDisposed) {
      throw StateError('Cannot borrow a disposed array.');
    }
    var minRelativeOffset = 0;
    if (!array.shape.contains(0)) {
      for (var d = 0; d < array.shape.length; d++) {
        final stride = array.strides[d];
        final size = array.shape[d];
        if (stride < 0) {
          minRelativeOffset += (size - 1) * stride;
        }
      }
    }
    final baseAddress =
        array.pointer.address + minRelativeOffset * array.dtype.byteWidth;
    return SendableNDArray._(
      mode: _SendableMode.borrow,
      address: baseAddress,
      shape: List<int>.unmodifiable(array.shape),
      strides: List<int>.unmodifiable(array.strides),
      dtypeIndex: array.dtype.index,
      physicalByteCapacity: array.physicalByteCapacity,
    );
  }

  /// The data type of the elements in the array.
  DType<T> get dtype => DType.values[dtypeIndex] as DType<T>;

  /// Returns true if this sendable array borrows shared native memory.
  bool get isBorrowed => _mode == _SendableMode.borrow;

  /// Returns true if this sendable array contains an independent data copy.
  bool get isCopy => _mode == _SendableMode.copy;

  /// Returns true if this [SendableNDArray] has already been materialized.
  bool get isMaterialized => _isMaterialized;

  /// The raw native C memory address if this is a borrowed array.
  ///
  /// It is an error if this [SendableNDArray] was created via [SendableNDArray.fromCopy].
  int get address {
    if (_address == null) {
      throw StateError(
        'Address is only available for borrowed SendableNDArray instances.',
      );
    }
    return _address;
  }

  /// The physical byte capacity of the backing buffer if this is a borrowed array.
  ///
  /// It is an error if this [SendableNDArray] was created via [SendableNDArray.fromCopy].
  int get physicalByteCapacity {
    if (_physicalByteCapacity == null) {
      throw StateError(
        'physicalByteCapacity is only available for borrowed SendableNDArray instances.',
      );
    }
    return _physicalByteCapacity;
  }

  /// Reconstructs an [NDArray<T>] from this [SendableNDArray].
  ///
  /// In copy mode ([SendableNDArray.fromCopy] or [NDArray.toSendable]), this allocates a new,
  /// scope-registered [NDArray<T>] on the current isolate's C heap and populates it from the
  /// transferred byte buffer.
  ///
  /// In borrow mode ([SendableNDArray.unsafeBorrow] or [NDArray.toSendableBorrow]), this delegates
  /// to [materializeView] to construct a zero-copy non-owning view over the borrowed address.
  ///
  /// It is an error if this [SendableNDArray] in copy mode has already been materialized,
  /// as [TransferableTypedData] can only be materialized once.
  ///
  /// **Performance considerations:**
  /// - In copy mode: $O(N)$ time and space complexity where $N$ is the number of elements.
  /// - In borrow mode: $O(1)$ time and space complexity.
  NDArray<T> materialize() {
    if (_mode == _SendableMode.borrow) {
      return materializeView();
    }
    if (_isMaterialized) {
      throw StateError(
        'SendableNDArray has already been materialized. '
        'TransferableTypedData can only be materialized once.',
      );
    }
    _isMaterialized = true;
    final Uint8List bytes;
    if (_isWasm) {
      bytes = _wasmBytes!;
    } else {
      final buffer = _transferableData!.materialize();
      bytes = buffer.asUint8List();
    }
    final result = NDArray<T>.create(shape, dtype);
    if (bytes.isNotEmpty) {
      final dst = result.pointer.cast<ffi.Uint8>();
      if (_isWasm) {
        for (var i = 0; i < bytes.length; i++) {
          dst[i] = bytes[i];
        }
      } else {
        dst.asTypedList(bytes.length).setAll(0, bytes);
      }
    }
    return result;
  }

  /// Reconstructs a non-owning zero-copy [NDArray<T>] view over the borrowed native memory address.
  ///
  /// The returned array shares the exact C memory of the source array and has no finalizer
  /// attached. Calling [NDArray.dispose] on it invalidates the view but will not free the
  /// underlying native C pointer.
  ///
  /// It is an error if this [SendableNDArray] was created via [SendableNDArray.fromCopy]
  /// rather than [SendableNDArray.unsafeBorrow].
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(1)$.
  /// - Space complexity: $O(1)$.
  NDArray<T> materializeView() {
    if (_mode != _SendableMode.borrow) {
      throw StateError(
        'Cannot materialize a view from a copied SendableNDArray. '
        'Use materialize() instead.',
      );
    }
    final ptr = ffi.Pointer<ffi.Void>.fromAddress(_address!);
    return NDArray<T>.fromPointer(
      ptr,
      shape,
      dtype,
      strides: strides,
      nativeFinalizer: null,
    );
  }
}
