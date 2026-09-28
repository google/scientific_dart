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
import 'dart:collection';

/// Bitwise conversion utilities and List wrappers for 16-bit floating point formats.
final class Float16Utils {
  Float16Utils._();

  static final _byteData = ByteData(8);

  /// Converts a 64-bit Dart [double] to a 16-bit IEEE 754 half-precision integer bit pattern.
  static int encodeFloat16(double value) {
    _byteData.setFloat64(0, value, Endian.little);
    final f64Bits = _byteData.getUint64(0, Endian.little);

    final sign = (f64Bits >> 63) & 0x1;
    final exp64 = ((f64Bits >> 52) & 0x7FF);
    final frac64 = f64Bits & 0xFFFFFFFFFFFFF;

    if (exp64 == 0x7FF) {
      // NaN or Infinity
      if (frac64 == 0) {
        return (sign << 15) | 0x7C00; // Infinity
      } else {
        return (sign << 15) | 0x7E00; // NaN
      }
    }

    if (exp64 == 0) {
      // Subnormal in float64 -> 0 in float16
      return sign << 15;
    }

    // Unbias float64 exponent (-1023) and rebias to float16 (+15)
    final exp16 = exp64 - 1023 + 15;

    if (exp16 >= 31) {
      // Overflow to infinity
      return (sign << 15) | 0x7C00;
    }

    if (exp16 <= 0) {
      // Subnormal in float16 or underflow to zero
      if (exp16 < -10) {
        return sign << 15; // Complete underflow
      }
      final fullFrac = frac64 | 0x10000000000000;
      final shift = 1 - exp16 + 42;
      final frac16 = (fullFrac >> shift) & 0x3FF;
      return (sign << 15) | frac16;
    }

    // Normal float16
    final frac16 = (frac64 >> 42) & 0x3FF;
    return (sign << 15) | (exp16 << 10) | frac16;
  }

  /// Converts a 16-bit IEEE 754 half-precision integer bit pattern to a Dart [double].
  static double decodeFloat16(int bits) {
    final sign = (bits >> 15) & 0x1;
    final exp16 = (bits >> 10) & 0x1F;
    final frac16 = bits & 0x3FF;

    if (exp16 == 0x1F) {
      if (frac16 == 0) {
        return sign == 1 ? double.negativeInfinity : double.infinity;
      } else {
        return double.nan;
      }
    }

    if (exp16 == 0) {
      if (frac16 == 0) {
        return sign == 1 ? -0.0 : 0.0;
      }
      // Subnormal in float16
      final val = frac16 / 1024.0 * 6.103515625e-5; // 2^-14
      return sign == 1 ? -val : val;
    }

    // Normal float16
    final exp64 = exp16 - 15 + 1023;
    final frac64 = frac16 << 42;
    final f64Bits = (sign << 63) | (exp64 << 52) | frac64;

    _byteData.setUint64(0, f64Bits, Endian.little);
    return _byteData.getFloat64(0, Endian.little);
  }

  /// Converts a 64-bit Dart [double] to a 16-bit BFloat16 integer bit pattern.
  static int encodeBFloat16(double value) {
    if (value.isNaN) {
      return 0x7FC0; // Standard quiet NaN for BFloat16
    }
    _byteData.setFloat32(0, value, Endian.little);
    final f32Bits = _byteData.getUint32(0, Endian.little);
    // Truncate to top 16 bits with round-to-nearest-even
    final lsb = (f32Bits >> 16) & 1;
    final roundingBias = 0x7FFF + lsb;
    final rounded = f32Bits + roundingBias;
    return (rounded >> 16) & 0xFFFF;
  }

  /// Converts a 16-bit BFloat16 integer bit pattern to a Dart [double].
  static double decodeBFloat16(int bits) {
    final f32Bits = bits << 16;
    _byteData.setUint32(0, f32Bits, Endian.little);
    return _byteData.getFloat32(0, Endian.little);
  }
}

/// A Dart [List] view wrapping a [Uint16List] containing IEEE 754 Float16 values.
final class Float16List with ListMixin<double> implements List<double> {
  final Uint16List _buffer;

  Float16List(this._buffer);

  @override
  int get length => _buffer.length;

  @override
  set length(int newLength) =>
      throw UnsupportedError('Cannot resize Float16List');

  @override
  double operator [](int index) => Float16Utils.decodeFloat16(_buffer[index]);

  @override
  void operator []=(int index, double value) {
    _buffer[index] = Float16Utils.encodeFloat16(value);
  }
}

/// A Dart [List] view wrapping a [Uint16List] containing BFloat16 values.
final class BFloat16List with ListMixin<double> implements List<double> {
  final Uint16List _buffer;

  BFloat16List(this._buffer);

  @override
  int get length => _buffer.length;

  @override
  set length(int newLength) =>
      throw UnsupportedError('Cannot resize BFloat16List');

  @override
  double operator [](int index) => Float16Utils.decodeBFloat16(_buffer[index]);

  @override
  void operator []=(int index, double value) {
    _buffer[index] = Float16Utils.encodeBFloat16(value);
  }
}
