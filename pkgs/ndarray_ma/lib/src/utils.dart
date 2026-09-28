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

part of 'masked_array.dart';

/// Resolves the common DType for binary operations.
DType _resolveDType(DType a, DType b) {
  if (a == DType.boolean && b == DType.boolean) return DType.uint8;
  if (a == DType.boolean) return b;
  if (b == DType.boolean) return a;
  if (a == b) return a;

  final isAIntLarge =
      a == DType.int64 ||
      a == DType.uint64 ||
      a == DType.int32 ||
      a == DType.uint32;
  final isBIntLarge =
      b == DType.int64 ||
      b == DType.uint64 ||
      b == DType.int32 ||
      b == DType.uint32;

  if (a == DType.complex128 || b == DType.complex128) return DType.complex128;
  if (a == DType.complex64 || b == DType.complex64) {
    if (a == DType.float64 || b == DType.float64) return DType.complex128;
    if (isAIntLarge || isBIntLarge) return DType.complex128;
    return DType.complex64;
  }
  if (a == DType.float64 || b == DType.float64) return DType.float64;
  if (a == DType.float32 || b == DType.float32) {
    if (isAIntLarge || isBIntLarge) return DType.float64;
    return DType.float32;
  }
  if (a == DType.float16 ||
      b == DType.float16 ||
      a == DType.bfloat16 ||
      b == DType.bfloat16) {
    if (a.isInteger || b.isInteger) return DType.float64;
    if (a != b) return DType.float32;
    return a;
  }
  if (a == DType.uint64 || b == DType.uint64) {
    if (a.isInteger &&
        b.isInteger &&
        (a != DType.uint64 || b != DType.uint64)) {
      if (a == DType.int64 ||
          a == DType.int32 ||
          a == DType.int16 ||
          a == DType.int8 ||
          b == DType.int64 ||
          b == DType.int32 ||
          b == DType.int16 ||
          b == DType.int8) {
        return DType.float64;
      }
    }
    return DType.uint64;
  }
  if (a == DType.int64 || b == DType.int64) return DType.int64;
  if (a == DType.uint32 || b == DType.uint32) {
    if (a == DType.int32 ||
        a == DType.int16 ||
        a == DType.int8 ||
        b == DType.int32 ||
        b == DType.int16 ||
        b == DType.int8) {
      return DType.int64;
    }
    return DType.uint32;
  }
  if (a == DType.int32 || b == DType.int32) return DType.int32;
  if (a == DType.uint16 || b == DType.uint16) {
    if (a == DType.int16 ||
        a == DType.int8 ||
        b == DType.int16 ||
        b == DType.int8) {
      return DType.int32;
    }
    return DType.uint16;
  }
  if (a == DType.int16 || b == DType.int16) return DType.int16;
  if (a == DType.uint8 || b == DType.uint8) {
    if (a == DType.int8 || b == DType.int8) return DType.int16;
    return DType.uint8;
  }
  return DType.int8;
}

/// Default fill value mapping based on DType.
Object _defaultFillValue(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return 1e20;
    case DType.complex128:
    case DType.complex64:
      return Complex(1e20, 0.0);
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      return 999999;
    case DType.boolean:
      return true;
  }
}

/// Coerces a scalar value to the correct Dart type for a given DType.
Object _coerceScalar(dynamic value, DType dtype) {
  if (dtype.isFloating) {
    return (value as num).toDouble();
  } else if (dtype.isComplex) {
    if (value is Complex) return value;
    return Complex((value as num).toDouble(), 0.0);
  } else if (dtype.isInteger) {
    return (value as num).toInt();
  } else if (dtype == DType.boolean) {
    if (value is bool) return value;
    return (value as num) != 0;
  }
  throw ArgumentError("Unsupported dtype: $dtype");
}

/// Helper to coerce or get default fill value.
Object? _coerceOrGetDefault(Object? value, DType dtype) {
  if (value == null) return null;
  try {
    return _coerceScalar(value, dtype);
  } catch (_) {
    return _defaultFillValue(dtype);
  }
}

/// Wraps a scalar value into a 0-dimensional NDArray of the given DType.
NDArray<S> _wrapScalar<S extends DTypeTag>(dynamic value, DType<S> dtype) {
  if (value is NDArray<S>) return value;
  final coerced = _coerceScalar(value, dtype);
  return NDArray<S>.fromList([coerced], [], dtype);
}

/// Helper to get max value for a DType, used in min reduction.
dynamic _maxValue(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return double.infinity;
    case DType.int64:
      return 9223372036854775807;
    case DType.int32:
      return 2147483647;
    case DType.int16:
      return 32767;
    case DType.int8:
      return 127;
    case DType.uint64:
      return -1;
    case DType.uint32:
      return 4294967295;
    case DType.uint16:
      return 65535;
    case DType.uint8:
      return 255;
    default:
      throw ArgumentError("No max value for dtype $dtype");
  }
}

/// Helper to get min value for a DType, used in max reduction.
dynamic _minValue(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return double.negativeInfinity;
    case DType.int64:
      return -9223372036854775808;
    case DType.int32:
      return -2147483648;
    case DType.int16:
      return -32768;
    case DType.int8:
      return -128;
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      return 0;
    default:
      throw ArgumentError("No min value for dtype $dtype");
  }
}

/// Dispatches binary operations to the correct generic implementation.
NDArray<DTypeTag> _dispatchBinary(
  NDArray<DTypeTag> a,
  NDArray<DTypeTag> b,
  String opName,
  DType<DTypeTag> targetDType,
) {
  switch (opName) {
    case 'add':
      return ndops.addAs(a, b, targetDType);
    case 'sub':
      return ndops.subtractAs(a, b, targetDType);
    case 'mul':
      return ndops.multiplyAs(a, b, targetDType);
    case 'div':
      final divDType = targetDType.isComplex ? DType.complex128 : DType.float64;
      return ndops.divideAs(a, b, divDType);
    default:
      throw ArgumentError("Unknown op: $opName");
  }
}

/// Dispatches MaskedArray creation to preserve runtime type parameter.
MaskedArray<DTypeTag> dispatchCreateMaskedArray(
  NDArray<DTypeTag> data,
  NDArray<Boolean> mask, {
  Object? fillValue,
}) {
  switch (data.dtype) {
    case DType.float64:
      return MaskedArray<Float64>(
        data as NDArray<Float64>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.float64),
      );
    case DType.float32:
      return MaskedArray<Float32>(
        data as NDArray<Float32>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.float32),
      );
    case DType.float16:
      return MaskedArray<Float16>(
        data as NDArray<Float16>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.float16),
      );
    case DType.bfloat16:
      return MaskedArray<BFloat16>(
        data as NDArray<BFloat16>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.bfloat16),
      );
    case DType.complex128:
      return MaskedArray<Complex128>(
        data as NDArray<Complex128>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.complex128),
      );
    case DType.complex64:
      return MaskedArray<Complex64>(
        data as NDArray<Complex64>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.complex64),
      );
    case DType.int64:
      return MaskedArray<Int64>(
        data as NDArray<Int64>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.int64),
      );
    case DType.int32:
      return MaskedArray<Int32>(
        data as NDArray<Int32>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.int32),
      );
    case DType.int16:
      return MaskedArray<Int16>(
        data as NDArray<Int16>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.int16),
      );
    case DType.int8:
      return MaskedArray<Int8>(
        data as NDArray<Int8>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.int8),
      );
    case DType.uint64:
      return MaskedArray<Uint64>(
        data as NDArray<Uint64>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.uint64),
      );
    case DType.uint32:
      return MaskedArray<Uint32>(
        data as NDArray<Uint32>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.uint32),
      );
    case DType.uint16:
      return MaskedArray<Uint16>(
        data as NDArray<Uint16>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.uint16),
      );
    case DType.uint8:
      return MaskedArray<Uint8>(
        data as NDArray<Uint8>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.uint8),
      );
    case DType.boolean:
      return MaskedArray<Boolean>(
        data as NDArray<Boolean>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.boolean),
      );
  }
}
