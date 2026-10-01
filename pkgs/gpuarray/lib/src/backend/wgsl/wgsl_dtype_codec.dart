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

import 'dart:typed_data';

import '../../dtype.dart';
import '../compute_engine.dart';

/// Universal 15-[DType] WebGPU Shading Language (WGSL) storage buffer and
/// compute code generator.
///
/// Generates WGSL buffer declarations, sub-32-bit atomic compare-and-swap
/// writers, 64-bit IEEE-754 and integer emulators, and 128-bit complex
/// arithmetic helpers so all 15 [DType]s execute natively on WebGPU compute
/// pipelines.
extension type const WgslDTypeCodec._(Object? _) {
  /// WGSL storage array element type for raw bit-preserving memory access.
  static String rawStorageElementType(DType dtype) => switch (dtype) {
    DType.float64 ||
    DType.int64 ||
    DType.uint64 ||
    DType.complex64 => 'vec2<u32>',
    DType.complex128 => 'vec4<u32>',
    DType.float32 ||
    DType.float16 ||
    DType.bfloat16 ||
    DType.int32 ||
    DType.int16 ||
    DType.int8 ||
    DType.uint32 ||
    DType.uint16 ||
    DType.uint8 ||
    DType.boolean => 'u32',
  };

  /// Whether [dtype] uses sub-32-bit packed storage requiring atomic writes.
  static bool isSubWord(DType dtype) => switch (dtype) {
    DType.float16 ||
    DType.bfloat16 ||
    DType.int16 ||
    DType.uint16 ||
    DType.int8 ||
    DType.uint8 ||
    DType.boolean => true,
    DType.float64 ||
    DType.float32 ||
    DType.int64 ||
    DType.int32 ||
    DType.uint64 ||
    DType.uint32 ||
    DType.complex64 ||
    DType.complex128 => false,
  };

  /// WGSL zero literal for the raw storage representation of [dtype].
  static String rawZeroLiteral(DType dtype) => switch (dtype) {
    DType.float64 ||
    DType.int64 ||
    DType.uint64 ||
    DType.complex64 => 'vec2<u32>(0u, 0u)',
    DType.complex128 => 'vec4<u32>(0u, 0u, 0u, 0u)',
    DType.float32 ||
    DType.float16 ||
    DType.bfloat16 ||
    DType.int32 ||
    DType.int16 ||
    DType.int8 ||
    DType.uint32 ||
    DType.uint16 ||
    DType.uint8 ||
    DType.boolean => '0u',
  };

  /// WGSL expression reconstructing a raw element of [dtype] from four `u32`
  /// uniform words ([word0], [word1], [word2], [word3]).
  static String rawFromUniformWords(
    DType dtype,
    String word0,
    String word1,
    String word2,
    String word3,
  ) => switch (dtype) {
    DType.complex128 => 'vec4<u32>($word0, $word1, $word2, $word3)',
    DType.float64 ||
    DType.int64 ||
    DType.uint64 ||
    DType.complex64 => 'vec2<u32>($word0, $word1)',
    DType.float32 ||
    DType.float16 ||
    DType.bfloat16 ||
    DType.int32 ||
    DType.int16 ||
    DType.int8 ||
    DType.uint32 ||
    DType.uint16 ||
    DType.uint8 ||
    DType.boolean => word0,
  };

  /// Packs a Dart scalar [value] into four 32-bit unsigned integer words
  /// matching the raw binary representation of [dtype].
  static List<int> packRawScalarWords(DType dtype, Object value) {
    final byteData = ByteData(16);
    switch (dtype) {
      case DType.float64:
        final number = _scalarToDouble(value);
        byteData.setFloat64(0, number, Endian.little);
      case DType.float32:
        final number = _scalarToDouble(value);
        byteData.setFloat32(0, number, Endian.little);
      case DType.float16:
        final number = _scalarToDouble(value);
        byteData.setUint32(
          0,
          ComputeEngine.doubleToFloat16Bits(number),
          Endian.little,
        );
      case DType.bfloat16:
        final number = _scalarToDouble(value);
        byteData.setUint32(
          0,
          ComputeEngine.doubleToBfloat16Bits(number),
          Endian.little,
        );
      case DType.int64:
      case DType.uint64:
        final integer = _scalarToInt(value);
        byteData.setInt64(0, integer, Endian.little);
      case DType.int32:
      case DType.uint32:
        final integer = _scalarToInt(value);
        byteData.setUint32(0, integer & 0xFFFFFFFF, Endian.little);
      case DType.int16:
      case DType.uint16:
        final integer = _scalarToInt(value);
        byteData.setUint32(0, integer & 0xFFFF, Endian.little);
      case DType.int8:
      case DType.uint8:
        final integer = _scalarToInt(value);
        byteData.setUint32(0, integer & 0xFF, Endian.little);
      case DType.boolean:
        final flag = _scalarToBool(value);
        byteData.setUint32(0, flag ? 1 : 0, Endian.little);
      case DType.complex64:
        final complex = _scalarToComplex(value);
        byteData.setFloat32(0, complex.real, Endian.little);
        byteData.setFloat32(4, complex.imag, Endian.little);
      case DType.complex128:
        final complex = _scalarToComplex(value);
        byteData.setFloat64(0, complex.real, Endian.little);
        byteData.setFloat64(8, complex.imag, Endian.little);
    }
    return <int>[
      byteData.getUint32(0, Endian.little),
      byteData.getUint32(4, Endian.little),
      byteData.getUint32(8, Endian.little),
      byteData.getUint32(12, Endian.little),
    ];
  }

  static double _scalarToDouble(Object value) {
    if (value is num) return value.toDouble();
    if (value is bool) return value ? 1.0 : 0.0;
    if (value is BigInt) return value.toDouble();
    if (value is Complex) return value.real;
    throw ArgumentError.value(
      value,
      'value',
      'Must be a numeric, boolean, or Complex value.',
    );
  }

  static int _scalarToInt(Object value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    if (value is bool) return value ? 1 : 0;
    if (value is BigInt) return value.toInt();
    if (value is Complex) return value.real.toInt();
    throw ArgumentError.value(
      value,
      'value',
      'Must be a numeric, boolean, or Complex value.',
    );
  }

  static bool _scalarToBool(Object value) {
    if (value is bool) return value;
    if (value is num) return value != 0;
    if (value is BigInt) return value != BigInt.zero;
    if (value is Complex) return value.real != 0.0 || value.imag != 0.0;
    throw ArgumentError.value(
      value,
      'value',
      'Must be a numeric, boolean, or Complex value.',
    );
  }

  static Complex _scalarToComplex(Object value) {
    if (value is Complex) return value;
    if (value is num) return Complex(value.toDouble(), 0.0);
    if (value is bool) return Complex(value ? 1.0 : 0.0, 0.0);
    if (value is BigInt) return Complex(value.toDouble(), 0.0);
    throw ArgumentError.value(
      value,
      'value',
      'Must be a numeric, boolean, or Complex value.',
    );
  }

  /// Generates a read-only storage buffer binding declaration for [dtype].
  static String readBindingDecl(int binding, String name, DType dtype) {
    final elemType = rawStorageElementType(dtype);
    return '@group(0) @binding($binding) var<storage, read> $name: array<$elemType>;';
  }

  /// Generates a read-write storage buffer binding declaration for [dtype].
  static String writeBindingDecl(int binding, String name, DType dtype) {
    if (isSubWord(dtype)) {
      return '@group(0) @binding($binding) var<storage, read_write> $name: array<atomic<u32>>;';
    }
    final elemType = rawStorageElementType(dtype);
    return '@group(0) @binding($binding) var<storage, read_write> $name: array<$elemType>;';
  }

  /// Generates WGSL helper functions `load_<name>(idx: u32)` for raw storage
  /// reads from [bufferName].
  static String rawLoadFunction(
    String functionName,
    String bufferName,
    DType dtype,
  ) {
    final elemType = rawStorageElementType(dtype);
    return switch (dtype) {
      DType.int8 || DType.uint8 || DType.boolean =>
        '''
fn $functionName(idx: u32) -> u32 {
  let word = $bufferName[idx >> 2u];
  let shift = (idx & 3u) * 8u;
  return (word >> shift) & 0xFFu;
}
''',
      DType.float16 || DType.bfloat16 || DType.int16 || DType.uint16 =>
        '''
fn $functionName(idx: u32) -> u32 {
  let word = $bufferName[idx >> 1u];
  let shift = (idx & 1u) * 16u;
  return (word >> shift) & 0xFFFFu;
}
''',
      DType.float64 ||
      DType.float32 ||
      DType.int64 ||
      DType.int32 ||
      DType.uint64 ||
      DType.uint32 ||
      DType.complex64 ||
      DType.complex128 =>
        '''
fn $functionName(idx: u32) -> $elemType {
  return $bufferName[idx];
}
''',
    };
  }

  /// Generates WGSL helper function `store_<name>(idx: u32, val: <Type>)` for
  /// raw storage writes into [bufferName].
  static String rawStoreFunction(
    String functionName,
    String bufferName,
    DType dtype,
  ) {
    final elemType = rawStorageElementType(dtype);
    return switch (dtype) {
      DType.int8 || DType.uint8 || DType.boolean =>
        '''
fn $functionName(idx: u32, val: u32) {
  let word_idx = idx >> 2u;
  let shift = (idx & 3u) * 8u;
  let clear_mask = ~(0xFFu << shift);
  let insert_bits = (val & 0xFFu) << shift;
  var old_word = atomicLoad(&$bufferName[word_idx]);
  loop {
    let new_word = (old_word & clear_mask) | insert_bits;
    let ex = atomicCompareExchangeWeak(&$bufferName[word_idx], old_word, new_word);
    if (ex.exchanged) {
      break;
    }
    old_word = ex.old_value;
  }
}
''',
      DType.float16 || DType.bfloat16 || DType.int16 || DType.uint16 =>
        '''
fn $functionName(idx: u32, val: u32) {
  let word_idx = idx >> 1u;
  let shift = (idx & 1u) * 16u;
  let clear_mask = ~(0xFFFFu << shift);
  let insert_bits = (val & 0xFFFFu) << shift;
  var old_word = atomicLoad(&$bufferName[word_idx]);
  loop {
    let new_word = (old_word & clear_mask) | insert_bits;
    let ex = atomicCompareExchangeWeak(&$bufferName[word_idx], old_word, new_word);
    if (ex.exchanged) {
      break;
    }
    old_word = ex.old_value;
  }
}
''',
      DType.float64 ||
      DType.float32 ||
      DType.int64 ||
      DType.int32 ||
      DType.uint64 ||
      DType.uint32 ||
      DType.complex64 ||
      DType.complex128 =>
        '''
fn $functionName(idx: u32, val: $elemType) {
  $bufferName[idx] = val;
}
''',
    };
  }

  /// WGSL value type used during arithmetic and reduction evaluation for [dtype].
  static String computeValueType(DType dtype) => switch (dtype) {
    DType.float64 || DType.float32 || DType.float16 || DType.bfloat16 => 'f32',
    DType.int32 || DType.int16 || DType.int8 => 'i32',
    DType.uint32 || DType.uint16 || DType.uint8 || DType.boolean => 'u32',
    DType.int64 || DType.uint64 => 'vec2<u32>',
    DType.complex64 || DType.complex128 => 'vec2<f32>',
  };

  /// Shared WGSL numeric conversion and 64-bit / complex helper functions.
  static const String wgslNumericHelpers = '''
fn f64_to_f32(bits: vec2<u32>) -> f32 {
  let lo = bits.x;
  let hi = bits.y;
  let sign = hi & 0x80000000u;
  let exp64 = (hi >> 20u) & 0x7FFu;
  let mant_hi = hi & 0xFFFFFu;
  if (exp64 == 0u) {
    return bitcast<f32>(sign);
  }
  if (exp64 == 0x7FFu) {
    let is_nan = (mant_hi != 0u) || (lo != 0u);
    return bitcast<f32>(sign | 0x7F800000u | select(0u, 0x400000u, is_nan));
  }
  let exp32 = i32(exp64) - 1023 + 127;
  if (exp32 >= 255) {
    return bitcast<f32>(sign | 0x7F800000u);
  }
  if (exp32 <= 0) {
    return bitcast<f32>(sign);
  }
  let mant23 = (mant_hi << 3u) | (lo >> 29u);
  let round_bit = (lo >> 28u) & 1u;
  let sticky = select(0u, 1u, (lo & 0x0FFFFFFFu) != 0u);
  let round_up = select(0u, 1u, (round_bit != 0u) && ((sticky != 0u) || ((mant23 & 1u) != 0u)));
  return bitcast<f32>(sign | ((u32(exp32) << 23u) + mant23 + round_up));
}

fn f32_to_f64(val: f32) -> vec2<u32> {
  let b = bitcast<u32>(val);
  let sign = b & 0x80000000u;
  let exp32 = (b >> 23u) & 0xFFu;
  let mant23 = b & 0x7FFFFFu;
  if (exp32 == 0u) {
    return vec2<u32>(0u, sign);
  }
  if (exp32 == 0xFFu) {
    return vec2<u32>(0u, sign | 0x7FF00000u | select(0u, 0x80000u, mant23 != 0u));
  }
  let exp64 = (exp32 - 127u + 1023u) & 0x7FFu;
  let hi = sign | (exp64 << 20u) | (mant23 >> 3u);
  let lo = (mant23 & 7u) << 29u;
  return vec2<u32>(lo, hi);
}

fn bf16_to_f32(bits: u32) -> f32 {
  return bitcast<f32>((bits & 0xFFFFu) << 16u);
}

fn f32_to_bf16(val: f32) -> u32 {
  let b = bitcast<u32>(val);
  if (((b & 0x7F800000u) == 0x7F800000u) && ((b & 0x7FFFFFu) != 0u)) {
    return 0x7FC0u;
  }
  return ((b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u) & 0xFFFFu;
}

fn f16_to_f32(bits: u32) -> f32 {
  return unpack2x16float(bits & 0xFFFFu).x;
}

fn f32_to_f16(val: f32) -> u32 {
  return pack2x16float(vec2<f32>(val, 0.0)) & 0xFFFFu;
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  let carry = select(0u, 1u, lo < a.x);
  let hi = a.y + b.y + carry;
  return vec2<u32>(lo, hi);
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let borrow = select(0u, 1u, a.x < b.x);
  let lo = a.x - b.x;
  let hi = a.y - b.y - borrow;
  return vec2<u32>(lo, hi);
}

fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let a_lo = a.x & 0xFFFFu;
  let a_hi = a.x >> 16u;
  let b_lo = b.x & 0xFFFFu;
  let b_hi = b.x >> 16u;
  let p0 = a_lo * b_lo;
  let p1 = a_lo * b_hi;
  let p2 = a_hi * b_lo;
  let p3 = a_hi * b_hi;
  let mid = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
  let lo = (p0 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
  let hi_carry = (mid >> 16u) + (p1 >> 16u) + (p2 >> 16u) + p3;
  let hi = hi_carry + a.x * b.y + a.y * b.x;
  return vec2<u32>(lo, hi);
}

fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  return (a.y < b.y) || ((a.y == b.y) && (a.x < b.x));
}

fn i64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  let ay = bitcast<i32>(a.y);
  let by = bitcast<i32>(b.y);
  return (ay < by) || ((a.y == b.y) && (a.x < b.x));
}

fn i64_neg(a: vec2<u32>) -> vec2<u32> {
  return u64_sub(vec2<u32>(0u, 0u), a);
}

fn i64_abs(a: vec2<u32>) -> vec2<u32> {
  return select(a, i64_neg(a), bitcast<i32>(a.y) < 0);
}

fn u64_to_f32(a: vec2<u32>) -> f32 {
  return f32(a.y) * 4294967296.0 + f32(a.x);
}

fn i64_to_f32(a: vec2<u32>) -> f32 {
  let is_neg = bitcast<i32>(a.y) < 0;
  let mag = select(a, i64_neg(a), is_neg);
  let f = u64_to_f32(mag);
  return select(f, -f, is_neg);
}

fn f32_to_u64(v: f32) -> vec2<u32> {
  if (v <= 0.0) {
    return vec2<u32>(0u, 0u);
  }
  let hi = u32(v / 4294967296.0);
  let lo = u32(max(0.0, v - f32(hi) * 4294967296.0));
  return vec2<u32>(lo, hi);
}

fn f32_to_i64(v: f32) -> vec2<u32> {
  if (v >= 0.0) {
    return f32_to_u64(v);
  }
  return i64_neg(f32_to_u64(-v));
}

fn cpx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cpx_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let denom = b.x * b.x + b.y * b.y;
  return vec2<f32>((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

fn cpx_pow(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let r = length(a);
  if (r == 0.0) {
    return vec2<f32>(0.0, 0.0);
  }
  let theta = atan2(a.y, a.x);
  let ln_r = log(r);
  let mag = exp(b.x * ln_r - b.y * theta);
  let ang = b.x * theta + b.y * ln_r;
  return vec2<f32>(mag * cos(ang), mag * sin(ang));
}

fn cpx_sqrt(a: vec2<f32>) -> vec2<f32> {
  let r = length(a);
  let re = sqrt(max(0.0, 0.5 * (r + a.x)));
  let im = select(-1.0, 1.0, a.y >= 0.0) * sqrt(max(0.0, 0.5 * (r - a.x)));
  return vec2<f32>(re, im);
}
''';

  /// Generates WGSL function `load_val_<name>(idx: u32) -> <ComputeType>`
  /// reading from [bufferName] using raw load function [rawLoadName].
  static String computeLoadFunction(
    String functionName,
    String rawLoadName,
    DType dtype,
  ) {
    final valType = computeValueType(dtype);
    final body = switch (dtype) {
      DType.float64 => 'return f64_to_f32($rawLoadName(idx));',
      DType.float32 => 'return bitcast<f32>($rawLoadName(idx));',
      DType.float16 => 'return f16_to_f32($rawLoadName(idx));',
      DType.bfloat16 => 'return bf16_to_f32($rawLoadName(idx));',
      DType.int32 => 'return bitcast<i32>($rawLoadName(idx));',
      DType.int16 =>
        'let raw = $rawLoadName(idx); return (i32(raw << 16u) >> 16u);',
      DType.int8 =>
        'let raw = $rawLoadName(idx); return (i32(raw << 24u) >> 24u);',
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 => 'return $rawLoadName(idx);',
      DType.boolean => 'return select(0u, 1u, $rawLoadName(idx) != 0u);',
      DType.int64 || DType.uint64 => 'return $rawLoadName(idx);',
      DType.complex64 =>
        'let raw = $rawLoadName(idx); return vec2<f32>(bitcast<f32>(raw.x), bitcast<f32>(raw.y));',
      DType.complex128 =>
        'let raw = $rawLoadName(idx); return vec2<f32>(f64_to_f32(raw.xy), f64_to_f32(raw.zw));',
    };
    return '''
fn $functionName(idx: u32) -> $valType {
  $body
}
''';
  }

  /// Generates WGSL function `store_val_<name>(idx: u32, val: <ComputeType>)`
  /// writing to [rawStoreName].
  static String computeStoreFunction(
    String functionName,
    String rawStoreName,
    DType dtype,
  ) {
    final valType = computeValueType(dtype);
    final body = switch (dtype) {
      DType.float64 => '$rawStoreName(idx, f32_to_f64(val));',
      DType.float32 => '$rawStoreName(idx, bitcast<u32>(val));',
      DType.float16 => '$rawStoreName(idx, f32_to_f16(val));',
      DType.bfloat16 => '$rawStoreName(idx, f32_to_bf16(val));',
      DType.int32 => '$rawStoreName(idx, bitcast<u32>(val));',
      DType.int16 => '$rawStoreName(idx, u32(val) & 0xFFFFu);',
      DType.int8 => '$rawStoreName(idx, u32(val) & 0xFFu);',
      DType.uint32 => '$rawStoreName(idx, val);',
      DType.uint16 => '$rawStoreName(idx, val & 0xFFFFu);',
      DType.uint8 => '$rawStoreName(idx, val & 0xFFu);',
      DType.boolean => '$rawStoreName(idx, select(0u, 1u, val != 0u));',
      DType.int64 || DType.uint64 => '$rawStoreName(idx, val);',
      DType.complex64 =>
        '$rawStoreName(idx, vec2<u32>(bitcast<u32>(val.x), bitcast<u32>(val.y)));',
      DType.complex128 =>
        '''
  let re64 = f32_to_f64(val.x);
  let im64 = f32_to_f64(val.y);
  $rawStoreName(idx, vec4<u32>(re64.x, re64.y, im64.x, im64.y));''',
    };
    return '''
fn $functionName(idx: u32, val: $valType) {
  $body
}
''';
  }

  /// Generates WGSL binary expression for [op] on [dtype] compute values `a`
  /// and `b`.
  static String binaryValueExpr(String op, DType dtype) {
    if (dtype.isComplex) {
      return switch (op) {
        'add' || '+' => 'a + b',
        'sub' || 'subtract' || '-' => 'a - b',
        'mul' || 'multiply' || '*' => 'cpx_mul(a, b)',
        'div' || 'divide' || '/' => 'cpx_div(a, b)',
        'pow' || 'power' => 'cpx_pow(a, b)',
        'max' || 'maximum' => 'select(b, a, length(a) >= length(b))',
        'min' || 'minimum' => 'select(b, a, length(a) <= length(b))',
        _ => 'a + b',
      };
    }
    if (dtype == DType.int64) {
      return switch (op) {
        'add' || '+' => 'u64_add(a, b)',
        'sub' || 'subtract' || '-' => 'u64_sub(a, b)',
        'mul' || 'multiply' || '*' => 'u64_mul(a, b)',
        'div' || 'divide' || '/' => 'f32_to_i64(i64_to_f32(a) / i64_to_f32(b))',
        'pow' || 'power' => 'f32_to_i64(pow(i64_to_f32(a), i64_to_f32(b)))',
        'rem' || 'remainder' || 'mod' || '%' =>
          'f32_to_i64(i64_to_f32(a) - floor(i64_to_f32(a) / i64_to_f32(b)) * i64_to_f32(b))',
        'max' || 'maximum' => 'select(a, b, i64_lt(a, b))',
        'min' || 'minimum' => 'select(b, a, i64_lt(a, b))',
        _ => 'u64_add(a, b)',
      };
    }
    if (dtype == DType.uint64) {
      return switch (op) {
        'add' || '+' => 'u64_add(a, b)',
        'sub' || 'subtract' || '-' => 'u64_sub(a, b)',
        'mul' || 'multiply' || '*' => 'u64_mul(a, b)',
        'div' || 'divide' || '/' => 'f32_to_u64(u64_to_f32(a) / u64_to_f32(b))',
        'pow' || 'power' => 'f32_to_u64(pow(u64_to_f32(a), u64_to_f32(b)))',
        'rem' || 'remainder' || 'mod' || '%' =>
          'f32_to_u64(u64_to_f32(a) - floor(u64_to_f32(a) / u64_to_f32(b)) * u64_to_f32(b))',
        'max' || 'maximum' => 'select(a, b, u64_lt(a, b))',
        'min' || 'minimum' => 'select(b, a, u64_lt(a, b))',
        _ => 'u64_add(a, b)',
      };
    }
    if (dtype == DType.boolean) {
      return switch (op) {
        'add' || '+' => 'select(0u, 1u, (a != 0u) || (b != 0u))',
        'sub' || 'subtract' || '-' => 'select(0u, 1u, a > b)',
        'mul' || 'multiply' || '*' => 'select(0u, 1u, (a != 0u) && (b != 0u))',
        'div' || 'divide' || '/' => 'select(0u, a / b, b != 0u)',
        'pow' || 'power' => 'u32(pow(f32(a), f32(b)))',
        'rem' || 'remainder' || 'mod' || '%' => 'select(0u, a % b, b != 0u)',
        'max' || 'maximum' => 'max(a, b)',
        'min' || 'minimum' => 'min(a, b)',
        _ => 'select(0u, 1u, (a != 0u) || (b != 0u))',
      };
    }
    final isSignedInt =
        dtype == DType.int32 || dtype == DType.int16 || dtype == DType.int8;
    final isUnsignedInt =
        dtype == DType.uint32 || dtype == DType.uint16 || dtype == DType.uint8;
    if (isSignedInt || isUnsignedInt) {
      final intType = isSignedInt ? 'i32' : 'u32';
      return switch (op) {
        'add' || '+' => 'a + b',
        'sub' || 'subtract' || '-' => 'a - b',
        'mul' || 'multiply' || '*' => 'a * b',
        'div' || 'divide' || '/' =>
          isSignedInt
              ? 'select(0, a / b, b != 0)'
              : 'select(0u, a / b, b != 0u)',
        'pow' || 'power' => '$intType(pow(f32(a), f32(b)))',
        'rem' || 'remainder' || 'mod' || '%' =>
          isSignedInt
              ? 'select(0, ((a % b) + b) % b, b != 0)'
              : 'select(0u, a % b, b != 0u)',
        'max' || 'maximum' => 'max(a, b)',
        'min' || 'minimum' => 'min(a, b)',
        _ => 'a + b',
      };
    }
    return switch (op) {
      'add' || '+' => 'a + b',
      'sub' || 'subtract' || '-' => 'a - b',
      'mul' || 'multiply' || '*' => 'a * b',
      'div' || 'divide' || '/' => 'a / b',
      'pow' || 'power' => 'pow(a, b)',
      'rem' || 'remainder' || 'mod' || '%' => 'a - floor(a / b) * b',
      'max' || 'maximum' => 'max(a, b)',
      'min' || 'minimum' => 'min(a, b)',
      _ => 'a + b',
    };
  }

  /// Generates WGSL boolean comparison expression (`bool`) for [op] on [dtype]
  /// compute values `a` and `b`.
  static String comparisonBoolExpr(String op, DType dtype) {
    if (dtype.isComplex) {
      return switch (op) {
        'eq' || 'equal' => '(a.x == b.x) && (a.y == b.y)',
        'ne' || 'not_equal' || 'notEqual' => '(a.x != b.x) || (a.y != b.y)',
        _ => '(a.x == b.x) && (a.y == b.y)',
      };
    }
    if (dtype == DType.int64) {
      return switch (op) {
        'eq' || 'equal' => '(a.x == b.x) && (a.y == b.y)',
        'ne' || 'not_equal' || 'notEqual' => '(a.x != b.x) || (a.y != b.y)',
        'lt' || 'less' => 'i64_lt(a, b)',
        'le' || 'less_equal' || 'lessEqual' => '!i64_lt(b, a)',
        'gt' || 'greater' => 'i64_lt(b, a)',
        'ge' || 'greater_equal' || 'greaterEqual' => '!i64_lt(a, b)',
        _ => '(a.x == b.x) && (a.y == b.y)',
      };
    }
    if (dtype == DType.uint64) {
      return switch (op) {
        'eq' || 'equal' => '(a.x == b.x) && (a.y == b.y)',
        'ne' || 'not_equal' || 'notEqual' => '(a.x != b.x) || (a.y != b.y)',
        'lt' || 'less' => 'u64_lt(a, b)',
        'le' || 'less_equal' || 'lessEqual' => '!u64_lt(b, a)',
        'gt' || 'greater' => 'u64_lt(b, a)',
        'ge' || 'greater_equal' || 'greaterEqual' => '!u64_lt(a, b)',
        _ => '(a.x == b.x) && (a.y == b.y)',
      };
    }
    return switch (op) {
      'eq' || 'equal' => 'a == b',
      'ne' || 'not_equal' || 'notEqual' => 'a != b',
      'lt' || 'less' => 'a < b',
      'le' || 'less_equal' || 'lessEqual' => 'a <= b',
      'gt' || 'greater' => 'a > b',
      'ge' || 'greater_equal' || 'greaterEqual' => 'a >= b',
      _ => 'a == b',
    };
  }

  /// Generates WGSL unary expression for [op] on [dtype] compute value `a`.
  static String unaryValueExpr(String op, DType dtype) {
    if (dtype.isComplex) {
      return switch (op) {
        'negate' || 'neg' => '-a',
        'abs' => 'vec2<f32>(length(a), 0.0)',
        'sqrt' => 'cpx_sqrt(a)',
        'rsqrt' => 'cpx_div(vec2<f32>(1.0, 0.0), cpx_sqrt(a))',
        'exp' => 'exp(a.x) * vec2<f32>(cos(a.y), sin(a.y))',
        'log' => 'vec2<f32>(log(length(a)), atan2(a.y, a.x))',
        'sin' => 'vec2<f32>(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y))',
        'cos' => 'vec2<f32>(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y))',
        'tan' =>
          'cpx_div(vec2<f32>(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y)), vec2<f32>(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y)))',
        'sinh' => 'vec2<f32>(sinh(a.x) * cos(a.y), cosh(a.x) * sin(a.y))',
        'cosh' => 'vec2<f32>(cosh(a.x) * cos(a.y), sinh(a.x) * sin(a.y))',
        'tanh' =>
          'cpx_div(vec2<f32>(sinh(a.x) * cos(a.y), cosh(a.x) * sin(a.y)), vec2<f32>(cosh(a.x) * cos(a.y), sinh(a.x) * sin(a.y)))',
        'floor' => 'vec2<f32>(floor(a.x), floor(a.y))',
        'ceil' => 'vec2<f32>(ceil(a.x), ceil(a.y))',
        'round' => 'vec2<f32>(round(a.x), round(a.y))',
        'relu' => 'vec2<f32>(max(a.x, 0.0), max(a.y, 0.0))',
        _ => 'vec2<f32>(${_float32UnaryExpr(op, 'a.x')}, a.y)',
      };
    }
    if (dtype == DType.int64) {
      return switch (op) {
        'negate' || 'neg' => 'i64_neg(a)',
        'abs' => 'i64_abs(a)',
        'floor' || 'ceil' || 'round' => 'a',
        'relu' => 'select(a, vec2<u32>(0u, 0u), bitcast<i32>(a.y) < 0)',
        _ => 'f32_to_i64(${_float32UnaryExpr(op, 'i64_to_f32(a)')})',
      };
    }
    if (dtype == DType.uint64) {
      return switch (op) {
        'negate' || 'neg' => 'i64_neg(a)',
        'abs' || 'floor' || 'ceil' || 'round' || 'relu' => 'a',
        _ => 'f32_to_u64(${_float32UnaryExpr(op, 'u64_to_f32(a)')})',
      };
    }
    if (dtype == DType.int32 || dtype == DType.int16 || dtype == DType.int8) {
      return switch (op) {
        'negate' || 'neg' => '-a',
        'abs' => 'abs(a)',
        'floor' || 'ceil' || 'round' => 'a',
        'relu' => 'max(a, 0)',
        'sign' => 'sign(a)',
        _ => 'i32(${_float32UnaryExpr(op, 'f32(a)')})',
      };
    }
    if (dtype == DType.uint32 ||
        dtype == DType.uint16 ||
        dtype == DType.uint8 ||
        dtype == DType.boolean) {
      return switch (op) {
        'negate' || 'neg' => '0u - a',
        'abs' || 'floor' || 'ceil' || 'round' || 'relu' => 'a',
        'sign' => 'select(0u, 1u, a > 0u)',
        _ => 'u32(${_float32UnaryExpr(op, 'f32(a)')})',
      };
    }
    return _float32UnaryExpr(op, 'a');
  }

  static String _float32UnaryExpr(String op, String arg) => switch (op) {
    'negate' || 'neg' => 'bitcast<f32>(bitcast<u32>($arg) ^ 0x80000000u)',
    'abs' => 'bitcast<f32>(bitcast<u32>($arg) & 0x7FFFFFFFu)',
    'sqrt' => 'sqrt($arg)',
    'rsqrt' => 'inverseSqrt($arg)',
    'exp' => 'exp($arg)',
    'log' => 'log($arg)',
    'sin' => 'sin($arg)',
    'cos' => 'cos($arg)',
    'tan' => 'tan($arg)',
    'asin' => 'asin($arg)',
    'acos' => 'acos($arg)',
    'atan' => 'atan($arg)',
    'sinh' => 'sinh($arg)',
    'cosh' => 'cosh($arg)',
    'tanh' => 'tanh($arg)',
    'floor' => 'floor($arg)',
    'ceil' => 'ceil($arg)',
    'round' => 'round($arg)',
    'sign' => 'sign($arg)',
    'relu' => 'max($arg, 0.0)',
    'sigmoid' => '(1.0 / (1.0 + exp(-($arg))))',
    'silu' => '(($arg) / (1.0 + exp(-($arg))))',
    'gelu' =>
      '(0.5 * ($arg) * (1.0 + tanh(0.7978845608 * (($arg) + 0.044715 * ($arg) * ($arg) * ($arg)))))',
    'hardswish' => '(($arg) * clamp(($arg) + 3.0, 0.0, 6.0) / 6.0)',
    'mish' => '(($arg) * tanh(log(1.0 + exp($arg))))',
    _ => arg,
  };
}
