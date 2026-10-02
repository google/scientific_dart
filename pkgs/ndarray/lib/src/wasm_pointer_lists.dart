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

import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'float16_utils.dart';
import 'ndarray.dart';

/// Whether the current Dart runtime is compiled via `dart2wasm`.
const bool isWasmRuntime = bool.fromEnvironment('dart.tool.dart2wasm');

/// Largest single native allocation, in bytes, on a 32-bit `wasm32` target.
///
/// `size_t` is 32 bits wide there and the Wasm `malloc` rejects requests above
/// `0x7fffffff`, so allocation paths check against this bound (when
/// `sizeOf<Size>() == 4`) rather than letting a byte count wrap. Callers that
/// round sizes up for alignment must leave their own headroom below it.
const int maxWasm32AllocationBytes = 0x7fffffff;

/// Copies [count] 64-bit integers from [ptr] into a new [Int64List].
///
/// On the VM this is a single bulk copy of the `asTypedList` view; on
/// `dart2wasm`, where `asTypedList` is unavailable, the elements are copied one
/// at a time.
Int64List copyInt64PointerToList(ffi.Pointer<ffi.Int64> ptr, int count) {
  if (!isWasmRuntime) {
    return Int64List.fromList(ptr.asTypedList(count));
  }
  final out = Int64List(count);
  for (var i = 0; i < count; i++) {
    out[i] = ptr[i];
  }
  return out;
}

/// Stores [value] at element [index] of pointer array [base] without using
/// `Pointer<Pointer<T>>.operator []=`, which calls `_abi()` on `dart2wasm`.
@pragma('vm:prefer-inline')
void setPointerAt<T extends ffi.NativeType>(
  ffi.Pointer<ffi.Pointer<T>> base,
  int index,
  ffi.Pointer<T> value,
) {
  if (ffi.sizeOf<ffi.Pointer<ffi.Void>>() == 4) {
    base.cast<ffi.Uint32>()[index] = value.address;
  } else {
    base.cast<ffi.Uint64>()[index] = value.address;
  }
}

/// A fixed-length `List<double>` view over an [ffi.Pointer<ffi.Double>].
final class WasmFloat64PointerList extends ListBase<double> {
  final ffi.Pointer<ffi.Double> _ptr;

  @override
  final int length;

  /// Creates a [WasmFloat64PointerList] backed by [_ptr] with [length] elements.
  WasmFloat64PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmFloat64PointerList');
  }

  @override
  double operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, double value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<double>` view over an [ffi.Pointer<ffi.Float>].
final class WasmFloat32PointerList extends ListBase<double> {
  final ffi.Pointer<ffi.Float> _ptr;

  @override
  final int length;

  /// Creates a [WasmFloat32PointerList] backed by [_ptr] with [length] elements.
  WasmFloat32PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmFloat32PointerList');
  }

  @override
  double operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, double value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Int64>].
final class WasmInt64PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Int64> _ptr;

  @override
  final int length;

  /// Creates a [WasmInt64PointerList] backed by [_ptr] with [length] elements.
  WasmInt64PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmInt64PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Uint64>].
final class WasmUint64PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Uint64> _ptr;

  @override
  final int length;

  /// Creates a [WasmUint64PointerList] backed by [_ptr] with [length] elements.
  WasmUint64PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmUint64PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Int32>].
final class WasmInt32PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Int32> _ptr;

  @override
  final int length;

  /// Creates a [WasmInt32PointerList] backed by [_ptr] with [length] elements.
  WasmInt32PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmInt32PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Uint32>].
final class WasmUint32PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Uint32> _ptr;

  @override
  final int length;

  /// Creates a [WasmUint32PointerList] backed by [_ptr] with [length] elements.
  WasmUint32PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmUint32PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Int16>].
final class WasmInt16PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Int16> _ptr;

  @override
  final int length;

  /// Creates a [WasmInt16PointerList] backed by [_ptr] with [length] elements.
  WasmInt16PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmInt16PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Uint16>].
final class WasmUint16PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Uint16> _ptr;

  @override
  final int length;

  /// Creates a [WasmUint16PointerList] backed by [_ptr] with [length] elements.
  WasmUint16PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmUint16PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Int8>].
final class WasmInt8PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Int8> _ptr;

  @override
  final int length;

  /// Creates a [WasmInt8PointerList] backed by [_ptr] with [length] elements.
  WasmInt8PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmInt8PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// A fixed-length `List<int>` view over an [ffi.Pointer<ffi.Uint8>].
final class WasmUint8PointerList extends ListBase<int> {
  final ffi.Pointer<ffi.Uint8> _ptr;

  @override
  final int length;

  /// Creates a [WasmUint8PointerList] backed by [_ptr] with [length] elements.
  WasmUint8PointerList(this._ptr, this.length);

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize WasmUint8PointerList');
  }

  @override
  int operator [](int index) {
    RangeError.checkValidIndex(index, this, 'index', length);
    return _ptr[index];
  }

  @override
  void operator []=(int index, int value) {
    RangeError.checkValidIndex(index, this, 'index', length);
    _ptr[index] = value;
  }
}

/// Creates a live Wasm pointer-backed data list view for [pointer], [dtype],
/// and element [count] without invoking `Pointer.asTypedList`.
List<dynamic> createWasmDataView(
  ffi.Pointer<ffi.Void> pointer,
  DType<DTypeTag> dtype,
  int count,
) {
  switch (dtype) {
    case DType.float64:
      return WasmFloat64PointerList(pointer.cast<ffi.Double>(), count);
    case DType.float32:
      return WasmFloat32PointerList(pointer.cast<ffi.Float>(), count);
    case DType.float16:
      return Float16List(
        WasmUint16PointerList(pointer.cast<ffi.Uint16>(), count),
      );
    case DType.bfloat16:
      return BFloat16List(
        WasmUint16PointerList(pointer.cast<ffi.Uint16>(), count),
      );
    case DType.int64:
      return WasmInt64PointerList(pointer.cast<ffi.Int64>(), count);
    case DType.uint64:
      return WasmUint64PointerList(pointer.cast<ffi.Uint64>(), count);
    case DType.int32:
      return WasmInt32PointerList(pointer.cast<ffi.Int32>(), count);
    case DType.uint32:
      return WasmUint32PointerList(pointer.cast<ffi.Uint32>(), count);
    case DType.int16:
      return WasmInt16PointerList(pointer.cast<ffi.Int16>(), count);
    case DType.uint16:
      return WasmUint16PointerList(pointer.cast<ffi.Uint16>(), count);
    case DType.int8:
      return WasmInt8PointerList(pointer.cast<ffi.Int8>(), count);
    case DType.uint8:
      return WasmUint8PointerList(pointer.cast<ffi.Uint8>(), count);
    case DType.complex128:
      return ComplexList(
        WasmFloat64PointerList(pointer.cast<ffi.Double>(), count * 2),
      );
    case DType.complex64:
      return ComplexList(
        WasmFloat32PointerList(pointer.cast<ffi.Float>(), count * 2),
      );
    case DType.boolean:
      return BoolList(WasmUint8PointerList(pointer.cast<ffi.Uint8>(), count));
  }
}
