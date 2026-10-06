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

import '../backend/wgsl/wgsl_types.dart';
import '../dtype.dart';

/// Encodes a Dart [double] into its exact IEEE-754 binary64 `(lo32, hi32)`
/// 32-bit unsigned integer words for WGSL uniforms.
(int, int) encodeDoubleFloatUniform(double value) {
  final byteData = ByteData(8)..setFloat64(0, value, Endian.little);
  final loBits = byteData.getUint32(0, Endian.little);
  final hiBits = byteData.getUint32(4, Endian.little);
  return (loBits, hiBits);
}

/// Core WGSL library for bit-exact 53-bit IEEE-754 `Float64` (`vec2<u32>`) and
/// `Complex128` (`vec4<u32>`) arithmetic using pure 32-bit integer instructions
/// (immune to shader compiler floating-point fast-math reassociation).
const String wgslDoubleFloatComplexLib = '''
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

fn u64_shl(a: vec2<u32>, shift: u32) -> vec2<u32> {
  if (shift == 0u) {
    return a;
  }
  if (shift < 32u) {
    return vec2<u32>(a.x << shift, (a.y << shift) | (a.x >> (32u - shift)));
  }
  if (shift < 64u) {
    return vec2<u32>(0u, a.x << (shift - 32u));
  }
  return vec2<u32>(0u, 0u);
}

fn u64_shr(a: vec2<u32>, shift: u32) -> vec2<u32> {
  if (shift == 0u) {
    return a;
  }
  if (shift < 32u) {
    return vec2<u32>((a.x >> shift) | (a.y << (32u - shift)), a.y >> shift);
  }
  if (shift < 64u) {
    return vec2<u32>(a.y >> (shift - 32u), 0u);
  }
  return vec2<u32>(0u, 0u);
}

fn mul32x32_64(a: u32, b: u32) -> vec2<u32> {
  let a_lo = a & 0xFFFFu;
  let a_hi = a >> 16u;
  let b_lo = b & 0xFFFFu;
  let b_hi = b >> 16u;
  let p0 = a_lo * b_lo;
  let p1 = a_lo * b_hi;
  let p2 = a_hi * b_lo;
  let p3 = a_hi * b_hi;
  let mid1 = p1 + (p0 >> 16u);
  let mid2 = p2 + (mid1 & 0xFFFFu);
  let lo = (p0 & 0xFFFFu) | ((mid2 & 0xFFFFu) << 16u);
  let hi = p3 + (mid1 >> 16u) + (mid2 >> 16u);
  return vec2<u32>(lo, hi);
}

fn f64_from_f32(val: f32) -> vec2<u32> {
  let bits = bitcast<u32>(val);
  let sign = bits & 0x80000000u;
  let exp32 = i32((bits >> 23u) & 0xFFu);
  let mant23 = bits & 0x007FFFFFu;
  if (exp32 == 0) {
    return vec2<u32>(0u, sign);
  }
  if (exp32 == 255) {
    let nan_bit = select(0u, 0x00080000u, mant23 != 0u);
    return vec2<u32>(0u, sign | 0x7FF00000u | nan_bit);
  }
  let exp64 = u32(exp32 - 127 + 1023);
  let hi = sign | (exp64 << 20u) | (mant23 >> 3u);
  let lo = (mant23 & 0x7u) << 29u;
  return vec2<u32>(lo, hi);
}

fn f64_from_i32(val: i32) -> vec2<u32> {
  if (val == 0) {
    return vec2<u32>(0u, 0u);
  }
  let sign = select(0u, 0x80000000u, val < 0);
  let mag = select(u32(val), 0u - u32(val), val < 0);
  let msb = 31u - countLeadingZeros(mag);
  let exp64 = (msb + 1023u) << 20u;
  let shifted = u64_shl(vec2<u32>(mag, 0u), 52u - msb);
  return vec2<u32>(shifted.x, sign | exp64 | (shifted.y & 0x000FFFFFu));
}

fn f64_from_u32(val: u32) -> vec2<u32> {
  if (val == 0u) {
    return vec2<u32>(0u, 0u);
  }
  let msb = 31u - countLeadingZeros(val);
  let exp64 = (msb + 1023u) << 20u;
  let shifted = u64_shl(vec2<u32>(val, 0u), 52u - msb);
  return vec2<u32>(shifted.x, exp64 | (shifted.y & 0x000FFFFFu));
}

fn f64_neg(a: vec2<u32>) -> vec2<u32> {
  if (((a.y & 0x7FFFFFFFu) | a.x) == 0u) {
    return vec2<u32>(0u, 0u);
  }
  return vec2<u32>(a.x, a.y ^ 0x80000000u);
}

fn f64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  var sign_a = a.y & 0x80000000u;
  var exp_a = i32((a.y >> 20u) & 0x7FFu);
  var ma = vec2<u32>(a.x, (a.y & 0x000FFFFFu) | 0x00100000u);

  var sign_b = b.y & 0x80000000u;
  var exp_b = i32((b.y >> 20u) & 0x7FFu);
  var mb = vec2<u32>(b.x, (b.y & 0x000FFFFFu) | 0x00100000u);

  if (exp_a == 0) {
    return b;
  }
  if (exp_b == 0) {
    return a;
  }
  if (exp_a == 2047) {
    return a;
  }
  if (exp_b == 2047) {
    return b;
  }

  let b_larger = (exp_b > exp_a) ||
      (exp_b == exp_a && (mb.y > ma.y || (mb.y == ma.y && mb.x > ma.x)));
  if (b_larger) {
    let ts = sign_a; sign_a = sign_b; sign_b = ts;
    let te = exp_a; exp_a = exp_b; exp_b = te;
    let tm = ma; ma = mb; mb = tm;
  }

  ma = u64_shl(ma, 3u);
  mb = u64_shl(mb, 3u);
  let diff = u32(exp_a - exp_b);
  if (diff >= 60u) {
    mb = vec2<u32>(0u, 0u);
  } else if (diff > 0u) {
    let shifted = u64_shr(mb, diff);
    let lost = u64_shl(vec2<u32>(1u, 0u), diff);
    let mask = u64_sub(lost, vec2<u32>(1u, 0u));
    let sticky = select(0u, 1u, ((mb.x & mask.x) | (mb.y & mask.y)) != 0u);
    mb = vec2<u32>(shifted.x | sticky, shifted.y);
  }

  var m = vec2<u32>(0u, 0u);
  var exp_res = exp_a;
  if (sign_a == sign_b) {
    m = u64_add(ma, mb);
    if ((m.y & 0x01000000u) != 0u) {
      let sticky = m.x & 1u;
      m = u64_shr(m, 1u);
      m.x = m.x | sticky;
      exp_res = exp_res + 1;
    }
  } else {
    m = u64_sub(ma, mb);
    if (m.x == 0u && m.y == 0u) {
      return vec2<u32>(0u, 0u);
    }
    for (var iter = 0u; iter < 60u; iter = iter + 1u) {
      if ((m.y & 0x00800000u) != 0u) {
        break;
      }
      m = u64_shl(m, 1u);
      exp_res = exp_res - 1;
    }
  }

  let guard = (m.x >> 2u) & 1u;
  let round_sticky = m.x & 3u;
  let lsb = (m.x >> 3u) & 1u;
  let round_up = select(0u, 1u, guard == 1u && (round_sticky != 0u || lsb == 1u));
  var sig = u64_add(u64_shr(m, 3u), vec2<u32>(round_up, 0u));
  if ((sig.y & 0x00200000u) != 0u) {
    sig = u64_shr(sig, 1u);
    exp_res = exp_res + 1;
  }
  if (exp_res <= 0) {
    return vec2<u32>(0u, 0u);
  }
  if (exp_res >= 2047) {
    return vec2<u32>(0u, sign_a | 0x7FF00000u);
  }
  return vec2<u32>(sig.x, sign_a | (u32(exp_res) << 20u) | (sig.y & 0x000FFFFFu));
}

fn f64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return f64_add(a, f64_neg(b));
}

fn f64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let sign = (a.y ^ b.y) & 0x80000000u;
  let exp_a = i32((a.y >> 20u) & 0x7FFu);
  let exp_b = i32((b.y >> 20u) & 0x7FFu);
  if (exp_a == 0 || exp_b == 0) {
    return vec2<u32>(0u, 0u);
  }
  if (exp_a == 2047 || exp_b == 2047) {
    return vec2<u32>(0u, sign | 0x7FF00000u);
  }
  let ma = vec2<u32>(a.x, (a.y & 0x000FFFFFu) | 0x00100000u);
  let mb = vec2<u32>(b.x, (b.y & 0x000FFFFFu) | 0x00100000u);

  let p00 = mul32x32_64(ma.x, mb.x);
  let p01 = mul32x32_64(ma.x, mb.y);
  let p10 = mul32x32_64(ma.y, mb.x);
  let p11 = mul32x32_64(ma.y, mb.y);

  let mid = u64_add(u64_add(p01, p10), vec2<u32>(p00.y, 0u));
  let top = u64_add(p11, vec2<u32>(mid.y, 0u));

  var exp_res = exp_a + exp_b - 1023;
  var s = 20u;
  if ((top.y & 0x200u) != 0u) {
    s = 21u;
    exp_res = exp_res + 1;
  }
  let res_lo = (mid.x >> s) | (top.x << (32u - s));
  let res_hi = (top.x >> s) | (top.y << (32u - s));
  let round_bit = (mid.x >> (s - 1u)) & 1u;
  var sig = u64_add(vec2<u32>(res_lo, res_hi), vec2<u32>(round_bit, 0u));
  if ((sig.y & 0x00200000u) != 0u) {
    sig = u64_shr(sig, 1u);
    exp_res = exp_res + 1;
  }
  if (exp_res <= 0) {
    return vec2<u32>(0u, 0u);
  }
  if (exp_res >= 2047) {
    return vec2<u32>(0u, sign | 0x7FF00000u);
  }
  return vec2<u32>(sig.x, sign | (u32(exp_res) << 20u) | (sig.y & 0x000FFFFFu));
}

fn f64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let sign = (a.y ^ b.y) & 0x80000000u;
  let exp_a = i32((a.y >> 20u) & 0x7FFu);
  let exp_b = i32((b.y >> 20u) & 0x7FFu);
  if (exp_a == 0) {
    return vec2<u32>(0u, 0u);
  }
  if (exp_b == 0) {
    return vec2<u32>(0u, sign | 0x7FF00000u);
  }
  let a_norm = vec2<u32>(a.x, (a.y & 0x000FFFFFu) | 0x3FF00000u);
  let b_norm = vec2<u32>(b.x, (b.y & 0x000FFFFFu) | 0x3FF00000u);
  let b_f32 = bitcast<f32>(0x3F800000u | ((b.y & 0x000FFFFFu) << 3u) | (b.x >> 29u));
  var y = f64_from_f32(1.0 / b_f32);
  let two = vec2<u32>(0u, 0x40000000u);
  y = f64_mul(y, f64_sub(two, f64_mul(b_norm, y)));
  y = f64_mul(y, f64_sub(two, f64_mul(b_norm, y)));
  var q = f64_mul(a_norm, y);
  let rem = f64_sub(a_norm, f64_mul(b_norm, q));
  q = f64_add(q, f64_mul(rem, y));

  let exp_q = i32((q.y >> 20u) & 0x7FFu);
  let exp_res = exp_q + (exp_a - exp_b);
  if (exp_res <= 0) {
    return vec2<u32>(0u, 0u);
  }
  if (exp_res >= 2047) {
    return vec2<u32>(0u, sign | 0x7FF00000u);
  }
  return vec2<u32>(q.x, sign | (u32(exp_res) << 20u) | (q.y & 0x000FFFFFu));
}

// Compatibility aliases so existing shaders work seamlessly on IEEE-754 f64/c128 bits.
fn unpack_f64_df(bits: vec2<u32>) -> vec2<u32> { return bits; }
fn pack_df_f64(df: vec2<u32>) -> vec2<u32> { return df; }
fn unpack_c128_cdf(bits: vec4<u32>) -> vec4<u32> { return bits; }
fn pack_cdf_c128(cdf: vec4<u32>) -> vec4<u32> { return cdf; }

fn df_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return f64_add(a, b); }
fn df_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return f64_sub(a, b); }
fn df_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return f64_mul(a, b); }
fn df_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return f64_div(a, b); }

fn cdf_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let re = f64_add(a.xy, b.xy);
  let im = f64_add(a.zw, b.zw);
  return vec4<u32>(re.x, re.y, im.x, im.y);
}

fn cdf_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let re = f64_sub(a.xy, b.xy);
  let im = f64_sub(a.zw, b.zw);
  return vec4<u32>(re.x, re.y, im.x, im.y);
}

fn cdf_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let ac = f64_mul(a.xy, b.xy);
  let bd = f64_mul(a.zw, b.zw);
  let ad = f64_mul(a.xy, b.zw);
  let bc = f64_mul(a.zw, b.xy);
  let re = f64_sub(ac, bd);
  let im = f64_add(ad, bc);
  return vec4<u32>(re.x, re.y, im.x, im.y);
}

fn cdf_scale(a: vec4<u32>, s: vec2<u32>) -> vec4<u32> {
  let re = f64_mul(a.xy, s);
  let im = f64_mul(a.zw, s);
  return vec4<u32>(re.x, re.y, im.x, im.y);
}

fn cdf_conj(a: vec4<u32>) -> vec4<u32> {
  let neg_im = f64_neg(a.zw);
  return vec4<u32>(a.x, a.y, neg_im.x, neg_im.y);
}

// Evaluates exp(sign_dir * 2 * pi * i * num_turns / den_turns) in 53-bit f64.
fn cdf_twiddle_ratio(num_turns: u32, den_turns: u32, sign_dir: f32) -> vec4<u32> {
  let f64_one = vec2<u32>(0u, 0x3FF00000u);
  let f64_zero = vec2<u32>(0u, 0u);
  if (num_turns == 0u || den_turns == 0u) {
    return vec4<u32>(f64_one.x, f64_one.y, f64_zero.x, f64_zero.y);
  }
  let rem_turns = num_turns % den_turns;
  if (rem_turns == 0u) {
    return vec4<u32>(f64_one.x, f64_one.y, f64_zero.x, f64_zero.y);
  }
  let q_num = 4u * rem_turns;
  // Nearest integer quadrant quad = round(q_num / den_turns)
  let quad = (q_num + (den_turns >> 1u)) / den_turns;
  let num_signed = i32(q_num) - i32(quad * den_turns);
  let r_quad = f64_div(f64_from_i32(num_signed), f64_from_u32(den_turns));

  // pi / 2 in IEEE-754 binary64: 0x3FF921FB54442D18
  let pi_over_2 = vec2<u32>(0x54442D18u, 0x3FF921FBu);
  let t = f64_mul(r_quad, pi_over_2);
  let t2 = f64_mul(t, t);

  // Cosine Taylor series on [-pi/4, pi/4] up to t^14 / 14!
  let c7 = f64_div(f64_from_i32(-1), vec2<u32>(0x4C3B2800u, 0x423438E1u)); // -1/14! = -1/87178291200
  let c6 = f64_div(f64_one, f64_from_u32(479001600u));
  let c5 = f64_div(f64_from_i32(-1), f64_from_u32(3628800u));
  let c4 = f64_div(f64_one, f64_from_u32(40320u));
  let c3 = f64_div(f64_from_i32(-1), f64_from_u32(720u));
  let c2 = f64_div(f64_one, f64_from_u32(24u));
  let c1 = vec2<u32>(0u, 0xBFE00000u); // -0.5
  var cos_t = f64_add(c6, f64_mul(t2, c7));
  cos_t = f64_add(c5, f64_mul(t2, cos_t));
  cos_t = f64_add(c4, f64_mul(t2, cos_t));
  cos_t = f64_add(c3, f64_mul(t2, cos_t));
  cos_t = f64_add(c2, f64_mul(t2, cos_t));
  cos_t = f64_add(c1, f64_mul(t2, cos_t));
  cos_t = f64_add(f64_one, f64_mul(t2, cos_t));

  // Sine Taylor series on [-pi/4, pi/4] up to t^13 / 13!
  // 13! = 6227020800 = 0x41F7328CC0000000
  let s6 = f64_div(f64_one, vec2<u32>(0xC0000000u, 0x41F7328Cu));
  let s5 = f64_div(f64_from_i32(-1), f64_from_u32(39916800u));
  let s4 = f64_div(f64_one, f64_from_u32(362880u));
  let s3 = f64_div(f64_from_i32(-1), f64_from_u32(5040u));
  let s2 = f64_div(f64_one, f64_from_u32(120u));
  let s1 = f64_div(f64_from_i32(-1), f64_from_u32(6u));
  var sin_poly = f64_add(s5, f64_mul(t2, s6));
  sin_poly = f64_add(s4, f64_mul(t2, sin_poly));
  sin_poly = f64_add(s3, f64_mul(t2, sin_poly));
  sin_poly = f64_add(s2, f64_mul(t2, sin_poly));
  sin_poly = f64_add(s1, f64_mul(t2, sin_poly));
  sin_poly = f64_add(f64_one, f64_mul(t2, sin_poly));
  let sin_t = f64_mul(t, sin_poly);

  let q_mod = quad & 3u;
  var re = cos_t;
  var im = sin_t;
  if (q_mod == 1u) {
    re = f64_neg(sin_t);
    im = cos_t;
  } else if (q_mod == 2u) {
    re = f64_neg(cos_t);
    im = f64_neg(sin_t);
  } else if (q_mod == 3u) {
    re = sin_t;
    im = f64_neg(cos_t);
  }
  if (sign_dir < 0.0) {
    im = f64_neg(im);
  }
  return vec4<u32>(re.x, re.y, im.x, im.y);
}
''';

String _wgslInputStorageType(DType dtype) => switch (dtype) {
  DType.float32 => 'f32',
  DType.float64 => 'vec2<u32>',
  DType.int32 => 'i32',
  DType.uint32 => 'u32',
  DType.int64 || DType.uint64 => 'vec2<u32>',
  DType.complex64 => 'vec2<f32>',
  DType.complex128 => 'vec4<u32>',
  DType.float16 ||
  DType.bfloat16 ||
  DType.int16 ||
  DType.uint16 ||
  DType.int8 ||
  DType.uint8 ||
  DType.boolean => 'u32',
};

String _wgslLoadInputAsCdf(
  DType dtype,
  String bufferName,
  String indexExpr,
) => switch (dtype) {
  DType.complex128 => '$bufferName[$indexExpr]',
  DType.complex64 =>
    'vec4<u32>(f64_from_f32($bufferName[$indexExpr].x), f64_from_f32($bufferName[$indexExpr].y))',
  DType.float64 => 'vec4<u32>($bufferName[$indexExpr], 0u, 0u)',
  DType.float32 => 'vec4<u32>(f64_from_f32($bufferName[$indexExpr]), 0u, 0u)',
  DType.int32 => 'vec4<u32>(f64_from_i32($bufferName[$indexExpr]), 0u, 0u)',
  DType.uint32 => 'vec4<u32>(f64_from_u32($bufferName[$indexExpr]), 0u, 0u)',
  DType.int64 =>
    'vec4<u32>(f64_add(f64_mul(f64_from_i32(bitcast<i32>($bufferName[$indexExpr].y)), vec2<u32>(0u, 0x41F00000u)), f64_from_u32($bufferName[$indexExpr].x)), 0u, 0u)',
  DType.uint64 =>
    'vec4<u32>(f64_add(f64_mul(f64_from_u32($bufferName[$indexExpr].y), vec2<u32>(0u, 0x41F00000u)), f64_from_u32($bufferName[$indexExpr].x)), 0u, 0u)',
  DType.float16 =>
    'vec4<u32>(f64_from_f32(unpack2x16float(($bufferName[($indexExpr) >> 1u] >> (((($indexExpr) & 1u) * 16u))) & 0xFFFFu).x), 0u, 0u)',
  DType.bfloat16 =>
    'vec4<u32>(f64_from_f32(bitcast<f32>((($bufferName[($indexExpr) >> 1u] >> (((($indexExpr) & 1u) * 16u))) & 0xFFFFu) << 16u)), 0u, 0u)',
  DType.int16 =>
    'vec4<u32>(f64_from_i32((bitcast<i32>($bufferName[($indexExpr) >> 1u] >> (((($indexExpr) & 1u) * 16u))) << 16) >> 16), 0u, 0u)',
  DType.uint16 =>
    'vec4<u32>(f64_from_u32(($bufferName[($indexExpr) >> 1u] >> (((($indexExpr) & 1u) * 16u))) & 0xFFFFu), 0u, 0u)',
  DType.int8 =>
    'vec4<u32>(f64_from_i32((bitcast<i32>($bufferName[($indexExpr) >> 2u] >> (((($indexExpr) & 3u) * 8u))) << 24) >> 24), 0u, 0u)',
  DType.uint8 || DType.boolean =>
    'vec4<u32>(f64_from_u32(($bufferName[($indexExpr) >> 2u] >> (((($indexExpr) & 3u) * 8u))) & 0xFFu), 0u, 0u)',
};

/// Core WGSL library for single-precision `Complex64` (`vec2<f32>`) arithmetic.
const String wgslComplex64Lib = '''
fn c64_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a + b;
}

fn c64_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b;
}

fn c64_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn c64_conj(a: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x, -a.y);
}

fn c64_twiddle_ratio(num_turns: u32, den_turns: u32, sign_dir: f32) -> vec2<f32> {
  if (num_turns == 0u || den_turns == 0u) {
    return vec2<f32>(1.0, 0.0);
  }
  let rem_turns = num_turns % den_turns;
  if (rem_turns == 0u) {
    return vec2<f32>(1.0, 0.0);
  }
  let angle = sign_dir * (6.283185307179586 * f32(rem_turns) / f32(den_turns));
  return vec2<f32>(cos(angle), sin(angle));
}
''';

String _wgslLoadInputAsC64(DType dtype, String bufferName, String indexExpr) =>
    switch (dtype) {
      DType.complex64 => '$bufferName[$indexExpr]',
      DType.float32 => 'vec2<f32>($bufferName[$indexExpr], 0.0)',
      _ => 'vec2<f32>(0.0, 0.0)',
    };

/// Builds a WGSL shader that gathers slices along an axis from an arbitrary-rank
/// strided input of [sourceDType] into a contiguous `[batchCount, workLength]`
/// buffer of `Complex128` (`vec4<u32>`) or `Complex64` (`vec2<f32>`), truncating
/// or zero-padding as needed.
WgslShaderModule buildFftGatherShader(
  DType sourceDType, {
  bool singlePrecision = false,
}) {
  final storageType = _wgslInputStorageType(sourceDType);
  if (singlePrecision) {
    final loadExpr = _wgslLoadInputAsC64(sourceDType, 'src_buf', 'phys_idx');
    final code =
        '''
struct GatherUniforms {
  batch_count: u32,
  work_length: u32,
  copy_length: u32,
  outer_rank: u32,
  src_offset: u32,
  axis_stride: i32,
  conjugate_input: u32,
  pad0: u32,
  outer_shape0: vec4<u32>,
  outer_shape1: vec4<u32>,
  outer_strides0: vec4<i32>,
  outer_strides1: vec4<i32>,
}

@group(0) @binding(0) var<storage, read> src_buf: array<$storageType>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: GatherUniforms;

fn get_outer_dim(d: u32) -> u32 {
  if (d < 4u) { return params.outer_shape0[d]; }
  return params.outer_shape1[d - 4u];
}

fn get_outer_stride(d: u32) -> i32 {
  if (d < 4u) { return params.outer_strides0[d]; }
  return params.outer_strides1[d - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.work_length;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.work_length;
  let elem_idx = linear_idx % params.work_length;
  if (elem_idx >= params.copy_length) {
    dst_buf[linear_idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  var rem = batch_idx;
  var base_offset = i32(params.src_offset);
  for (var i = 0u; i < params.outer_rank; i = i + 1u) {
    let d = params.outer_rank - 1u - i;
    let dim_size = get_outer_dim(d);
    let coord = rem % dim_size;
    rem = rem / dim_size;
    base_offset = base_offset + i32(coord) * get_outer_stride(d);
  }
  let phys_idx = u32(base_offset + i32(elem_idx) * params.axis_stride);
  var val = $loadExpr;
  if (params.conjugate_input != 0u) {
    val = vec2<f32>(val.x, -val.y);
  }
  dst_buf[linear_idx] = val;
}
''';
    return WgslShaderModule(
      code: code,
      entryPoint: 'main',
      name: 'fft_gather_f32_${sourceDType.name}',
    );
  }

  final loadExpr = _wgslLoadInputAsCdf(sourceDType, 'src_buf', 'phys_idx');
  final code =
      '''
$wgslDoubleFloatComplexLib

struct GatherUniforms {
  batch_count: u32,
  work_length: u32,
  copy_length: u32,
  outer_rank: u32,
  src_offset: u32,
  axis_stride: i32,
  conjugate_input: u32,
  pad0: u32,
  outer_shape0: vec4<u32>,
  outer_shape1: vec4<u32>,
  outer_strides0: vec4<i32>,
  outer_strides1: vec4<i32>,
}

@group(0) @binding(0) var<storage, read> src_buf: array<$storageType>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: GatherUniforms;

fn get_outer_dim(d: u32) -> u32 {
  if (d < 4u) { return params.outer_shape0[d]; }
  return params.outer_shape1[d - 4u];
}

fn get_outer_stride(d: u32) -> i32 {
  if (d < 4u) { return params.outer_strides0[d]; }
  return params.outer_strides1[d - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.work_length;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.work_length;
  let elem_idx = linear_idx % params.work_length;
  if (elem_idx >= params.copy_length) {
    dst_buf[linear_idx] = vec4<u32>(0u, 0u, 0u, 0u);
    return;
  }
  var rem = batch_idx;
  var base_offset = i32(params.src_offset);
  for (var i = 0u; i < params.outer_rank; i = i + 1u) {
    let d = params.outer_rank - 1u - i;
    let dim_size = get_outer_dim(d);
    let coord = rem % dim_size;
    rem = rem / dim_size;
    base_offset = base_offset + i32(coord) * get_outer_stride(d);
  }
  let phys_idx = u32(base_offset + i32(elem_idx) * params.axis_stride);
  var val_cdf = $loadExpr;
  if (params.conjugate_input != 0u) {
    val_cdf = cdf_conj(val_cdf);
  }
  dst_buf[linear_idx] = pack_cdf_c128(val_cdf);
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_gather_${sourceDType.name}',
  );
}

/// Builds a WGSL shader that expands `[batchCount, inputBins]` into a full
/// Hermitian-symmetric spectrum `[batchCount, targetN]` for `irfft` / `hfft`.
WgslShaderModule buildFftHermitianExpandShader({bool singlePrecision = false}) {
  if (singlePrecision) {
    const code = '''
struct HermitianUniforms {
  batch_count: u32,
  input_bins: u32,
  target_n: u32,
  half_n: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: HermitianUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.target_n;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.target_n;
  let k = linear_idx % params.target_n;
  let copy_bins = min(params.input_bins, params.half_n);
  if (k < copy_bins) {
    dst_buf[linear_idx] = src_buf[batch_idx * params.input_bins + k];
  } else if (k >= params.half_n) {
    let mirror = params.target_n - k;
    if (mirror < copy_bins) {
      let m_val = src_buf[batch_idx * params.input_bins + mirror];
      dst_buf[linear_idx] = vec2<f32>(m_val.x, -m_val.y);
    } else {
      dst_buf[linear_idx] = vec2<f32>(0.0, 0.0);
    }
  } else {
    dst_buf[linear_idx] = vec2<f32>(0.0, 0.0);
  }
}
''';
    return WgslShaderModule(
      code: code,
      entryPoint: 'main',
      name: 'fft_hermitian_expand_f32',
    );
  }
  const code =
      '''
$wgslDoubleFloatComplexLib

struct HermitianUniforms {
  batch_count: u32,
  input_bins: u32,
  target_n: u32,
  half_n: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: HermitianUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.target_n;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.target_n;
  let k = linear_idx % params.target_n;
  let copy_bins = min(params.input_bins, params.half_n);
  if (k < copy_bins) {
    dst_buf[linear_idx] = src_buf[batch_idx * params.input_bins + k];
  } else if (k >= params.half_n) {
    let mirror = params.target_n - k;
    if (mirror < copy_bins) {
      let m_val = unpack_c128_cdf(src_buf[batch_idx * params.input_bins + mirror]);
      dst_buf[linear_idx] = pack_cdf_c128(cdf_conj(m_val));
    } else {
      dst_buf[linear_idx] = vec4<u32>(0u, 0u, 0u, 0u);
    }
  } else {
    dst_buf[linear_idx] = vec4<u32>(0u, 0u, 0u, 0u);
  }
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_hermitian_expand',
  );
}

/// Builds a WGSL shader that bit-reverses indices along each row of `[batchCount, n]`.
WgslShaderModule buildFftBitReverseShader({bool singlePrecision = false}) {
  final elemType = singlePrecision ? 'vec2<f32>' : 'vec4<u32>';
  final code =
      '''
struct BitRevUniforms {
  batch_count: u32,
  n: u32,
  log2_n: u32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<$elemType>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<$elemType>;
@group(0) @binding(2) var<uniform> params: BitRevUniforms;

fn bit_reverse(v: u32, bits: u32) -> u32 {
  if (bits == 0u) {
    return 0u;
  }
  return reverseBits(v) >> (32u - bits);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.n;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.n;
  let k = linear_idx % params.n;
  let rev_k = bit_reverse(k, params.log2_n);
  dst_buf[linear_idx] = src_buf[batch_idx * params.n + rev_k];
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: singlePrecision ? 'fft_bit_reverse_f32' : 'fft_bit_reverse',
  );
}

/// Builds a WGSL shader that executes one Cooley-Tukey radix-2 butterfly stage
/// across `[batchCount, n]` in Complex64 or 53-bit IEEE-754 Complex128 precision.
WgslShaderModule buildFftButterflyStageShader({bool singlePrecision = false}) {
  if (singlePrecision) {
    const code =
        '''
$wgslComplex64Lib

struct ButterflyUniforms {
  batch_count: u32,
  n: u32,
  half_span: u32,
  sign_bits: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: ButterflyUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairs_per_row = params.n >> 1u;
  let total_pairs = params.batch_count * pairs_per_row;
  let pair_idx = gid.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  let batch_idx = pair_idx / pairs_per_row;
  let within_row = pair_idx % pairs_per_row;
  let span = params.half_span << 1u;
  let group_idx = within_row / params.half_span;
  let j = within_row % params.half_span;
  let row_base = batch_idx * params.n;
  let even_idx = row_base + group_idx * span + j;
  let odd_idx = even_idx + params.half_span;

  let sign_dir = bitcast<f32>(params.sign_bits);
  let twiddle = c64_twiddle_ratio(j, span, sign_dir);
  let u = src_buf[even_idx];
  let v = c64_mul(src_buf[odd_idx], twiddle);

  dst_buf[even_idx] = c64_add(u, v);
  dst_buf[odd_idx] = c64_sub(u, v);
}
''';
    return WgslShaderModule(
      code: code,
      entryPoint: 'main',
      name: 'fft_butterfly_stage_f32',
    );
  }
  const code =
      '''
$wgslDoubleFloatComplexLib

struct ButterflyUniforms {
  batch_count: u32,
  n: u32,
  half_span: u32,
  sign_bits: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: ButterflyUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairs_per_row = params.n >> 1u;
  let total_pairs = params.batch_count * pairs_per_row;
  let pair_idx = gid.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  let batch_idx = pair_idx / pairs_per_row;
  let within_row = pair_idx % pairs_per_row;
  let span = params.half_span << 1u;
  let group_idx = within_row / params.half_span;
  let j = within_row % params.half_span;
  let row_base = batch_idx * params.n;
  let even_idx = row_base + group_idx * span + j;
  let odd_idx = even_idx + params.half_span;

  let sign_dir = bitcast<f32>(params.sign_bits);
  let twiddle = cdf_twiddle_ratio(j, span, sign_dir);
  let u = unpack_c128_cdf(src_buf[even_idx]);
  let v = cdf_mul(unpack_c128_cdf(src_buf[odd_idx]), twiddle);

  dst_buf[even_idx] = pack_cdf_c128(cdf_add(u, v));
  dst_buf[odd_idx] = pack_cdf_c128(cdf_sub(u, v));
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_butterfly_stage',
  );
}

/// Builds a WGSL shader that computes a direct 1D DFT along each row of
/// `[batchCount, n]` for arbitrary non-power-of-2 lengths in Complex64 or Complex128 precision.
WgslShaderModule buildFftDirectDftShader({bool singlePrecision = false}) {
  if (singlePrecision) {
    const code =
        '''
$wgslComplex64Lib

struct DirectDftUniforms {
  batch_count: u32,
  n: u32,
  sign_bits: u32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: DirectDftUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.n;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.n;
  let k = linear_idx % params.n;
  let row_base = batch_idx * params.n;
  let sign_dir = bitcast<f32>(params.sign_bits);

  var acc = vec2<f32>(0.0, 0.0);
  for (var m = 0u; m < params.n; m = m + 1u) {
    let x_m = src_buf[row_base + m];
    let turn_num = (m * k) % params.n;
    let w = c64_twiddle_ratio(turn_num, params.n, sign_dir);
    acc = c64_add(acc, c64_mul(x_m, w));
  }
  dst_buf[linear_idx] = acc;
}
''';
    return WgslShaderModule(
      code: code,
      entryPoint: 'main',
      name: 'fft_direct_dft_f32',
    );
  }
  const code =
      '''
$wgslDoubleFloatComplexLib

struct DirectDftUniforms {
  batch_count: u32,
  n: u32,
  sign_bits: u32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: DirectDftUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.n;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.n;
  let k = linear_idx % params.n;
  let row_base = batch_idx * params.n;
  let sign_dir = bitcast<f32>(params.sign_bits);

  var acc = vec4<u32>(0u, 0u, 0u, 0u);
  for (var m = 0u; m < params.n; m = m + 1u) {
    let x_m = unpack_c128_cdf(src_buf[row_base + m]);
    let turn_num = (m * k) % params.n;
    let w = cdf_twiddle_ratio(turn_num, params.n, sign_dir);
    acc = cdf_add(acc, cdf_mul(x_m, w));
  }
  dst_buf[linear_idx] = pack_cdf_c128(acc);
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_direct_dft',
  );
}
