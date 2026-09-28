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

import '../ndarray.dart';
import '../ndarray_bindings.dart';

/// Defensive typed-pointer accessors for passing [NDArray] buffers across the
/// Dart-to-native FFI boundary.
extension NDArrayNativePointer on NDArray {
  /// Returns [pointer] cast to `ffi.Pointer<N>` after verifying that [dtype]
  /// is layout-compatible with the native element type [N].
  ///
  /// Throws a [StateError] if [isDisposed] is true or if [dtype] does not match
  /// the expected native element type [N].
  ffi.Pointer<N> typedPointer<N extends ffi.NativeType>() {
    if (!_matchesNativeType<N>(dtype)) {
      throw StateError(
        'Native pointer type mismatch: kernel expects Pointer<$N>, '
        'but array has dtype $dtype (${dtype.byteWidth} bytes).',
      );
    }
    return pointer.cast<N>();
  }

  /// Returns [pointer] cast to `ffi.Pointer<N>` for C kernels that represent
  /// [DType.complex128] as interleaved `double*` or [DType.complex64] as
  /// interleaved `float*`.
  ///
  /// Throws a [StateError] if [isDisposed] is true or if [dtype] is not a
  /// complex dtype matching [N].
  ffi.Pointer<N> complexComponentPointer<N extends ffi.NativeType>() {
    final valid =
        (N == ffi.Double && dtype == DType.complex128) ||
        (N == ffi.Float && dtype == DType.complex64);
    if (!valid) {
      throw StateError(
        'Complex component pointer mismatch: kernel expects Pointer<$N>, '
        'but array has dtype $dtype.',
      );
    }
    return pointer.cast<N>();
  }
}

bool _matchesNativeType<N extends ffi.NativeType>(DType dtype) {
  if (N == ffi.Void || N == ffi.NativeType) {
    return true;
  }
  return switch (dtype) {
    DType.float64 => N == ffi.Double,
    DType.float32 => N == ffi.Float,
    DType.float16 ||
    DType.bfloat16 ||
    DType.uint16 => N == ffi.Uint16 || N == ffi.UnsignedShort,
    DType.int64 => N == ffi.Int64 || N == ffi.LongLong,
    DType.uint64 => N == ffi.Uint64 || N == ffi.UnsignedLongLong,
    DType.int32 => N == ffi.Int32 || N == ffi.Int,
    DType.uint32 => N == ffi.Uint32 || N == ffi.UnsignedInt,
    DType.int16 => N == ffi.Int16 || N == ffi.Short,
    DType.int8 => N == ffi.Int8 || N == ffi.SignedChar,
    DType.uint8 ||
    DType.boolean => N == ffi.Uint8 || N == ffi.UnsignedChar || N == ffi.Bool,
    DType.complex128 => N == cpx_t,
    DType.complex64 => N == cpx_f_t,
  };
}
