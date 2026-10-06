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

import '../backend/wgsl/wgsl_types.dart';
import '../dtype.dart';

final Map<String, WgslShaderModule> _linalgShaderCache =
    <String, WgslShaderModule>{};

/// Retrieves or compiles a cached [WgslShaderModule] for [name].
WgslShaderModule getOrCreateLinalgShader(
  String name,
  String Function() buildCode, {
  int workgroupSize = 1,
}) {
  return _linalgShaderCache.putIfAbsent(
    name,
    () => WgslShaderModule(
      name: name,
      code: buildCode(),
      workgroupSize: WgslWorkgroupSize(workgroupSize),
    ),
  );
}

/// Whether [dtype] represents a complex floating-point type.
bool isComplexDType(DType dtype) {
  switch (dtype) {
    case DType.complex64:
    case DType.complex128:
      return true;
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
      return false;
  }
}

/// Whether [dtype] is a 32-bit single-precision real or complex floating-point
/// type ([DType.float32] or [DType.complex64]).
bool isSinglePrecisionDType(DType dtype) {
  switch (dtype) {
    case DType.float32:
    case DType.complex64:
      return true;
    case DType.float64:
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
      return false;
  }
}

/// Projects [dtype] to its inexact math `M` output dtype (`Float32 -> Float32`,
/// `Complex64 -> Complex64`, `Complex128 -> Complex128`, others -> `Float64`).
DType linalgMathDType(DType dtype) {
  switch (dtype) {
    case DType.float32:
      return DType.float32;
    case DType.complex64:
      return DType.complex64;
    case DType.complex128:
      return DType.complex128;
    case DType.float64:
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
      return DType.float64;
  }
}

/// Projects [dtype] to its real floating-point computation `F` output dtype
/// (`Float32`/`Complex64 -> Float32`, others -> `Float64`).
DType linalgFloatDType(DType dtype) {
  switch (dtype) {
    case DType.float32:
    case DType.complex64:
      return DType.float32;
    case DType.float64:
    case DType.complex128:
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
      return DType.float64;
  }
}

/// Projects [dtype] to its complex `C` output dtype
/// (`Float32`/`Complex64 -> Complex64`, others -> `Complex128`).
DType linalgComplexDType(DType dtype) {
  switch (dtype) {
    case DType.float32:
    case DType.complex64:
      return DType.complex64;
    case DType.float64:
    case DType.complex128:
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
      return DType.complex128;
  }
}

/// High-precision Dekker/Knuth double-float (`df64` as `vec2<f32>`) and
/// complex double-float (`cdf64` as `vec4<f32>`) WGSL library with IEEE-754
/// binary64 (`vec2<u32>`) pack/unpack routines.
const String linalgDf64WgslLibrary = r'''
fn force_f32(x: f32) -> f32 {
  let u = bitcast<u32>(x);
  let z = (countOneBits(u) + countLeadingZeros(u)) >> 6u;
  return bitcast<f32>(u | z);
}

fn df64_zero() -> vec2<f32> {
  return vec2<f32>(0.0, 0.0);
}

fn df64_one() -> vec2<f32> {
  return vec2<f32>(1.0, 0.0);
}

fn df64_from_f32(x: f32) -> vec2<f32> {
  return vec2<f32>(x, 0.0);
}

fn df64_neg(a: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(-a.x, -a.y);
}

fn df64_abs(a: vec2<f32>) -> vec2<f32> {
  if (a.x < 0.0 || (a.x == 0.0 && a.y < 0.0)) {
    return vec2<f32>(-a.x, -a.y);
  }
  return a;
}

fn df64_quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = force_f32(a + b);
  let t = force_f32(s - a);
  let e = force_f32(b - t);
  return vec2<f32>(s, e);
}

fn df64_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = force_f32(a + b);
  let v = force_f32(s - a);
  let e = force_f32(force_f32(a - force_f32(s - v)) + force_f32(b - v));
  return vec2<f32>(s, e);
}

fn df64_two_prod(a: f32, b: f32) -> vec2<f32> {
  let p = force_f32(a * b);
  let a_hi = bitcast<f32>(bitcast<u32>(a) & 0xFFFFF000u);
  let a_lo = force_f32(a - a_hi);
  let b_hi = bitcast<f32>(bitcast<u32>(b) & 0xFFFFF000u);
  let b_lo = force_f32(b - b_hi);
  let e = force_f32(force_f32(force_f32(force_f32(a_hi * b_hi - p) + force_f32(a_hi * b_lo)) + force_f32(a_lo * b_hi)) + force_f32(a_lo * b_lo));
  return vec2<f32>(p, e);
}

fn df64_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let s = df64_two_sum(a.x, b.x);
  let t = df64_two_sum(a.y, b.y);
  let c = df64_quick_two_sum(s.x, force_f32(s.y + t.x));
  return df64_quick_two_sum(c.x, force_f32(c.y + t.y));
}

fn df64_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return df64_add(a, df64_neg(b));
}

fn df64_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let p = df64_two_prod(a.x, b.x);
  let cross = force_f32(force_f32(a.x * b.y) + force_f32(a.y * b.x));
  return df64_quick_two_sum(p.x, force_f32(p.y + cross));
}

fn df64_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  if (b.x == 0.0 && b.y == 0.0) {
    let sign_a = (bitcast<u32>(a.x) ^ bitcast<u32>(b.x)) & 0x80000000u;
    if (a.x == 0.0 && a.y == 0.0) {
      let nan_f = bitcast<f32>(0x7FC00000u);
      return vec2<f32>(nan_f, nan_f);
    }
    return vec2<f32>(bitcast<f32>(sign_a | 0x7F800000u), 0.0);
  }
  let q1 = force_f32(a.x / b.x);
  let r1 = df64_sub(a, df64_mul(vec2<f32>(q1, 0.0), b));
  let q2 = force_f32(r1.x / b.x);
  let r2 = df64_sub(r1, df64_mul(vec2<f32>(q2, 0.0), b));
  let q3 = force_f32(r2.x / b.x);
  let s1 = df64_quick_two_sum(q1, q2);
  return df64_add(s1, vec2<f32>(q3, 0.0));
}

fn df64_sqrt(a: vec2<f32>) -> vec2<f32> {
  if (a.x <= 0.0 && a.y <= 0.0) {
    if (a.x < 0.0) {
      let nan_f = bitcast<f32>(0x7FC00000u);
      return vec2<f32>(nan_f, nan_f);
    }
    return vec2<f32>(0.0, 0.0);
  }
  let x0 = force_f32(sqrt(a.x));
  let sq = df64_two_prod(x0, x0);
  let rem = df64_sub(a, sq);
  let dx = force_f32(rem.x / force_f32(2.0 * x0));
  let x1 = df64_quick_two_sum(x0, dx);
  let sq2 = df64_mul(x1, x1);
  let rem2 = df64_sub(a, sq2);
  let dx2 = force_f32(rem2.x / force_f32(2.0 * x1.x));
  return df64_add(x1, vec2<f32>(dx2, 0.0));
}

fn df64_log(a: vec2<f32>) -> vec2<f32> {
  if (a.x <= 0.0) {
    if (a.x == 0.0 && a.y == 0.0) {
      return vec2<f32>(bitcast<f32>(0xFF800000u), 0.0);
    }
    let nan_f = bitcast<f32>(0x7FC00000u);
    return vec2<f32>(nan_f, nan_f);
  }
  let bits = bitcast<u32>(a.x);
  var exp_i = i32((bits >> 23u) & 0xFFu) - 127;
  let scale = bitcast<f32>(u32(127 - exp_i) << 23u);
  var m = vec2<f32>(a.x * scale, a.y * scale);
  if (m.x > 1.41421356) {
    m = vec2<f32>(m.x * 0.5, m.y * 0.5);
    exp_i = exp_i + 1;
  }
  let one = vec2<f32>(1.0, 0.0);
  let z = df64_div(df64_sub(m, one), df64_add(m, one));
  let z2 = df64_mul(z, z);
  var term = z;
  var acc = z;
  for (var k = 1; k <= 8; k = k + 1) {
    term = df64_mul(term, z2);
    let denom = vec2<f32>(f32(2 * k + 1), 0.0);
    acc = df64_add(acc, df64_div(term, denom));
  }
  let ln_m = vec2<f32>(acc.x * 2.0, acc.y * 2.0);
  let ln2 = vec2<f32>(0.6931471824645996, -1.9046543e-9);
  return df64_add(ln_m, df64_mul(vec2<f32>(f32(exp_i), 0.0), ln2));
}

fn df64_lt(a: vec2<f32>, b: vec2<f32>) -> bool {
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

fn df64_le(a: vec2<f32>, b: vec2<f32>) -> bool {
  return (a.x < b.x) || (a.x == b.x && a.y <= b.y);
}

fn df64_gt(a: vec2<f32>, b: vec2<f32>) -> bool {
  return (a.x > b.x) || (a.x == b.x && a.y > b.y);
}

fn df64_ge(a: vec2<f32>, b: vec2<f32>) -> bool {
  return (a.x > b.x) || (a.x == b.x && a.y >= b.y);
}

fn df64_eq(a: vec2<f32>, b: vec2<f32>) -> bool {
  return a.x == b.x && a.y == b.y;
}

fn df64_max(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  if (df64_gt(a, b)) {
    return a;
  }
  return b;
}

fn df64_min(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  if (df64_lt(a, b)) {
    return a;
  }
  return b;
}

fn unpack_f64_df64(bits: vec2<u32>) -> vec2<f32> {
  let sign_bit = bits.y & 0x80000000u;
  let exp64 = (bits.y >> 20u) & 0x7FFu;
  let mant_hi = bits.y & 0x000FFFFFu;
  let mant_lo = bits.x;
  if (exp64 == 0u) {
    return vec2<f32>(bitcast<f32>(sign_bit), 0.0);
  }
  if (exp64 == 0x7FFu) {
    if (mant_hi == 0u && mant_lo == 0u) {
      return vec2<f32>(bitcast<f32>(sign_bit | 0x7F800000u), 0.0);
    }
    let nan_f = bitcast<f32>(sign_bit | 0x7FC00000u);
    return vec2<f32>(nan_f, nan_f);
  }
  let exp32 = i32(exp64) - 1023 + 127;
  if (exp32 >= 255) {
    return vec2<f32>(bitcast<f32>(sign_bit | 0x7F800000u), 0.0);
  }
  if (exp32 <= 0) {
    return vec2<f32>(bitcast<f32>(sign_bit), 0.0);
  }
  let mant23 = (mant_hi << 3u) | (mant_lo >> 29u);
  let val_hi = bitcast<f32>(sign_bit | (u32(exp32) << 23u) | mant23);
  let rem29 = mant_lo & 0x1FFFFFFFu;
  if (rem29 == 0u || exp32 <= 29) {
    return vec2<f32>(val_hi, 0.0);
  }
  if (exp32 > 52) {
    let scale_hi = bitcast<f32>(sign_bit | (u32(exp32 - 44) << 23u));
    let scale_lo = bitcast<f32>(sign_bit | (u32(exp32 - 52) << 23u));
    let lo_a = f32(rem29 >> 8u) * scale_hi;
    let lo_b = f32(rem29 & 0xFFu) * scale_lo;
    let s1 = df64_quick_two_sum(val_hi, lo_a);
    return df64_quick_two_sum(s1.x, force_f32(s1.y + lo_b));
  }
  let scale_mid = bitcast<f32>(sign_bit | (u32(exp32 - 29) << 23u));
  let val_lo = (f32(rem29) * (1.0 / 536870912.0)) * scale_mid;
  return df64_quick_two_sum(val_hi, val_lo);
}

fn pack_df64_f64(v_in: vec2<f32>) -> vec2<u32> {
  let v = df64_quick_two_sum(v_in.x, v_in.y);
  let hi_bits = bitcast<u32>(v.x);
  let sign_bit = hi_bits & 0x80000000u;
  let exp32 = (hi_bits >> 23u) & 0xFFu;
  let mant23 = hi_bits & 0x007FFFFFu;
  if (exp32 == 0u) {
    return vec2<u32>(0u, sign_bit);
  }
  if (exp32 == 0xFFu) {
    if (mant23 == 0u) {
      return vec2<u32>(0u, sign_bit | 0x7FF00000u);
    }
    return vec2<u32>(0u, sign_bit | 0x7FF80000u);
  }
  var full_hi = 0x00100000u | (mant23 >> 3u);
  var full_lo = (mant23 & 7u) << 29u;
  var final_exp64 = exp32 + 896u;

  let lo_bits = bitcast<u32>(v.y);
  let lo_sign = lo_bits & 0x80000000u;
  let lo_exp32 = (lo_bits >> 23u) & 0xFFu;
  let lo_mant23 = lo_bits & 0x007FFFFFu;
  var delta = 0u;
  if (lo_exp32 > 0u && lo_exp32 < 255u) {
    let diff = i32(exp32) - i32(lo_exp32);
    let lo_sig24 = 0x00800000u | lo_mant23;
    if (diff <= 29 && diff >= 23) {
      delta = lo_sig24 << u32(29 - diff);
    } else if (diff > 29 && diff <= 53) {
      let shift_r = u32(diff - 29);
      let half = 1u << (shift_r - 1u);
      delta = (lo_sig24 + half) >> shift_r;
    }
  }
  if (delta != 0u) {
    if (lo_sign == sign_bit) {
      let next_lo = full_lo + delta;
      let carry = select(0u, 1u, next_lo < full_lo);
      full_lo = next_lo;
      full_hi = full_hi + carry;
      if (full_hi >= 0x00200000u) {
        full_lo = (full_lo >> 1u) | ((full_hi & 1u) << 31u);
        full_hi = full_hi >> 1u;
        final_exp64 = final_exp64 + 1u;
      }
    } else {
      let borrow = select(0u, 1u, full_lo < delta);
      full_lo = full_lo - delta;
      full_hi = full_hi - borrow;
      if (full_hi < 0x00100000u) {
        full_hi = (full_hi << 1u) | (full_lo >> 31u);
        full_lo = full_lo << 1u;
        final_exp64 = final_exp64 - 1u;
      }
    }
  }
  let out_hi = sign_bit | (final_exp64 << 20u) | (full_hi & 0x000FFFFFu);
  return vec2<u32>(full_lo, out_hi);
}

fn cdf64_zero() -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn cdf64_one() -> vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

fn cdf64_from_df64(re: vec2<f32>, im: vec2<f32>) -> vec4<f32> {
  return vec4<f32>(re.x, re.y, im.x, im.y);
}

fn cdf64_re(z: vec4<f32>) -> vec2<f32> {
  return vec2<f32>(z.x, z.y);
}

fn cdf64_im(z: vec4<f32>) -> vec2<f32> {
  return vec2<f32>(z.z, z.w);
}

fn cdf64_conj(z: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(z.x, z.y, -z.z, -z.w);
}

fn cdf64_neg(z: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(-z.x, -z.y, -z.z, -z.w);
}

fn cdf64_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let re = df64_add(vec2<f32>(a.x, a.y), vec2<f32>(b.x, b.y));
  let im = df64_add(vec2<f32>(a.z, a.w), vec2<f32>(b.z, b.w));
  return vec4<f32>(re.x, re.y, im.x, im.y);
}

fn cdf64_sub(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let re = df64_sub(vec2<f32>(a.x, a.y), vec2<f32>(b.x, b.y));
  let im = df64_sub(vec2<f32>(a.z, a.w), vec2<f32>(b.z, b.w));
  return vec4<f32>(re.x, re.y, im.x, im.y);
}

fn cdf64_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let ar = vec2<f32>(a.x, a.y);
  let ai = vec2<f32>(a.z, a.w);
  let br = vec2<f32>(b.x, b.y);
  let bi = vec2<f32>(b.z, b.w);
  let re = df64_sub(df64_mul(ar, br), df64_mul(ai, bi));
  let im = df64_add(df64_mul(ar, bi), df64_mul(ai, br));
  return vec4<f32>(re.x, re.y, im.x, im.y);
}

fn cdf64_scale(z: vec4<f32>, s: vec2<f32>) -> vec4<f32> {
  let re = df64_mul(vec2<f32>(z.x, z.y), s);
  let im = df64_mul(vec2<f32>(z.z, z.w), s);
  return vec4<f32>(re.x, re.y, im.x, im.y);
}

fn cdf64_abs2(z: vec4<f32>) -> vec2<f32> {
  let zr = vec2<f32>(z.x, z.y);
  let zi = vec2<f32>(z.z, z.w);
  return df64_add(df64_mul(zr, zr), df64_mul(zi, zi));
}

fn cdf64_abs(z: vec4<f32>) -> vec2<f32> {
  return df64_sqrt(cdf64_abs2(z));
}

fn cdf64_div(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let denom = cdf64_abs2(b);
  let num = cdf64_mul(a, cdf64_conj(b));
  let re = df64_div(vec2<f32>(num.x, num.y), denom);
  let im = df64_div(vec2<f32>(num.z, num.w), denom);
  return vec4<f32>(re.x, re.y, im.x, im.y);
}
''';
