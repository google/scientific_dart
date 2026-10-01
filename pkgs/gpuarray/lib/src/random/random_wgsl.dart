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
import '../fft/fft_wgsl.dart';

/// Core WGSL library for `Philox4x32-10` counter-based stateless PRNG and
/// strided output coordinate mapping.
const String wgslPhilox4x32Lib =
    '''
$wgslDoubleFloatComplexLib

const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;

fn philox_round(c: vec4<u32>, k: vec2<u32>) -> vec4<u32> {
  let p0 = mul32x32_64(PHILOX_M0, c.x);
  let p1 = mul32x32_64(PHILOX_M1, c.z);
  return vec4<u32>(p1.y ^ c.y ^ k.x, p1.x, p0.y ^ c.w ^ k.y, p0.x);
}

fn philox4x32_10(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  var c = counter;
  var k = key;
  for (var r = 0u; r < 10u; r = r + 1u) {
    c = philox_round(c, k);
    k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
  }
  return c;
}

fn philox_counter_offset(base_c: vec4<u32>, elem_idx: u32, sub_idx: u32) -> vec4<u32> {
  let c0 = base_c.x + elem_idx;
  let carry0 = select(0u, 1u, c0 < base_c.x);
  let c1 = base_c.y + carry0;
  let carry1 = select(0u, 1u, c1 < base_c.y);
  let c2 = base_c.z + sub_idx + carry1;
  let carry2 = select(0u, 1u, c2 < base_c.z);
  let c3 = base_c.w + carry2;
  return vec4<u32>(c0, c1, c2, c3);
}

// Converts two 32-bit random words into a 53-bit uniform Float64 in [0, 1).
fn philox_u01_f64(w0: u32, w1: u32) -> vec2<u32> {
  let one_plus_u = vec2<u32>((w0 << 20u) | (w1 >> 12u), 0x3FF00000u | (w0 >> 12u));
  let f64_one = vec2<u32>(0u, 0x3FF00000u);
  return f64_sub(one_plus_u, f64_one);
}

// Converts a 32-bit random word into an open-interval (0, 1) f32.
fn philox_u01_open_f32(w: u32) -> f32 {
  return (f32(w >> 8u) + 0.5) * (1.0 / 16777216.0);
}

// Standard normal pair via Box-Muller transform in f32.
fn philox_box_muller(w0: u32, w1: u32) -> vec2<f32> {
  let u1 = philox_u01_open_f32(w0);
  let u2 = philox_u01_open_f32(w1);
  let radius = sqrt(-2.0 * log(u1));
  let theta = 6.283185307179586 * u2;
  return vec2<f32>(radius * cos(theta), radius * sin(theta));
}
''';

const String _wgslOutLayoutStructAndHelpers = '''
struct RngUniforms {
  total_elements: u32,
  rank: u32,
  out_offset: u32,
  mode: u32,
  key0: u32,
  key1: u32,
  counter0: u32,
  counter1: u32,
  counter2: u32,
  counter3: u32,
  param0_lo: u32,
  param0_hi: u32,
  param1_lo: u32,
  param1_hi: u32,
  param2_lo: u32,
  param2_hi: u32,
  shape0: vec4<u32>,
  shape1: vec4<u32>,
  strides0: vec4<i32>,
  strides1: vec4<i32>,
}

fn get_dim(p: RngUniforms, d: u32) -> u32 {
  if (d < 4u) { return p.shape0[d]; }
  return p.shape1[d - 4u];
}

fn get_stride(p: RngUniforms, d: u32) -> i32 {
  if (d < 4u) { return p.strides0[d]; }
  return p.strides1[d - 4u];
}

fn compute_out_phys_idx(p: RngUniforms, linear_idx: u32) -> u32 {
  if (p.rank == 0u) {
    return p.out_offset;
  }
  var rem = linear_idx;
  var off = i32(p.out_offset);
  for (var i = 0u; i < p.rank; i = i + 1u) {
    let d = p.rank - 1u - i;
    let dim_size = get_dim(p, d);
    let coord = rem % dim_size;
    rem = rem / dim_size;
    off = off + i32(coord) * get_stride(p, d);
  }
  return u32(off);
}
''';

/// Builds a WGSL compute shader for continuous `Float64` random distributions:
/// - `mode == 0`: `uniform(low, high)` (where `param0 = low`, `param1 = high - low`)
/// - `mode == 1`: `normal(loc, scale)` (where `param0 = loc`, `param1 = scale`)
/// - `mode == 2`: `exponential(scale)` (where `param0 = scale`)
/// - `mode == 3`: `bernoulli(p)` (where `param0 = p`)
/// - `mode == 4`: `truncatedNormal(low, high, loc, scale)` (`param0 = (low_f32, high_f32)`, `param1 = loc`, `param2 = scale`)
/// - `mode == 5`: `gamma(alpha, scale)` (`param0 = alpha_f32`, `param1 = scale`)
/// - `mode == 6`: `beta(a, b)` (`param0 = a_f32`, `param1 = b_f32`)
WgslShaderModule buildRandomFloat64Shader() {
  const code =
      '''
$wgslPhilox4x32Lib
$_wgslOutLayoutStructAndHelpers

@group(0) @binding(0) var<storage, read_write> dst_buf: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> params: RngUniforms;

// Marsaglia and Tsang's method for Gamma(alpha, 1.0) in WGSL.
fn sample_gamma_unit(alpha_in: f32, base_c: vec4<u32>, key: vec2<u32>, elem_idx: u32, stream_base: u32) -> f32 {
  let boost = alpha_in < 1.0;
  let alpha = select(alpha_in, alpha_in + 1.0, boost);
  let d = alpha - (1.0 / 3.0);
  let c = 1.0 / sqrt(9.0 * d);
  var sample_val = d;

  for (var iter = 0u; iter < 64u; iter = iter + 1u) {
    let ctr = philox_counter_offset(base_c, elem_idx, stream_base + iter);
    let blk = philox4x32_10(ctr, key);
    let z = philox_box_muller(blk.x, blk.y).x;
    let v_root = 1.0 + c * z;
    if (v_root > 0.0) {
      let v = v_root * v_root * v_root;
      let u = philox_u01_open_f32(blk.z);
      let z2 = z * z;
      if (u < 1.0 - 0.0331 * z2 * z2 || log(u) < 0.5 * z2 + d * (1.0 - v + log(v))) {
        sample_val = d * v;
        break;
      }
    }
  }
  if (boost) {
    let ctr_u = philox_counter_offset(base_c, elem_idx, stream_base + 64u);
    let blk_u = philox4x32_10(ctr_u, key);
    let u_boost = philox_u01_open_f32(blk_u.x);
    sample_val = sample_val * pow(u_boost, 1.0 / alpha_in);
  }
  return max(sample_val, 1e-30);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total_elements) {
    return;
  }
  let base_c = vec4<u32>(params.counter0, params.counter1, params.counter2, params.counter3);
  let key = vec2<u32>(params.key0, params.key1);
  let phys_idx = compute_out_phys_idx(params, idx);
  let p0 = vec2<u32>(params.param0_lo, params.param0_hi);
  let p1 = vec2<u32>(params.param1_lo, params.param1_hi);
  let p2 = vec2<u32>(params.param2_lo, params.param2_hi);

  if (params.mode == 0u) {
    // uniform: low + u * range
    let blk = philox4x32_10(philox_counter_offset(base_c, idx, 0u), key);
    let u = philox_u01_f64(blk.x, blk.y);
    dst_buf[phys_idx] = f64_add(p0, f64_mul(u, p1));
  } else if (params.mode == 1u) {
    // normal: loc + scale * z
    let blk = philox4x32_10(philox_counter_offset(base_c, idx, 0u), key);
    let z = philox_box_muller(blk.x, blk.y).x;
    dst_buf[phys_idx] = f64_add(p0, f64_mul(p1, f64_from_f32(z)));
  } else if (params.mode == 2u) {
    // exponential: -scale * ln(u)
    let blk = philox4x32_10(philox_counter_offset(base_c, idx, 0u), key);
    let u = philox_u01_open_f32(blk.x);
    let exp_sample = -log(u);
    dst_buf[phys_idx] = f64_mul(p0, f64_from_f32(exp_sample));
  } else if (params.mode == 3u) {
    // bernoulli: u < p ? 1.0 : 0.0
    let blk = philox4x32_10(philox_counter_offset(base_c, idx, 0u), key);
    let u = philox_u01_open_f32(blk.x);
    let prob = bitcast<f32>(params.param0_lo);
    let one = vec2<u32>(0u, 0x3FF00000u);
    let zero = vec2<u32>(0u, 0u);
    dst_buf[phys_idx] = select(zero, one, u < prob);
  } else if (params.mode == 4u) {
    // truncatedNormal: z in [low, high], output = loc + scale * z
    let low_z = bitcast<f32>(params.param0_lo);
    let high_z = bitcast<f32>(params.param0_hi);
    var chosen_z = 0.5 * (low_z + high_z);
    var accepted = false;
    for (var iter = 0u; iter < 64u; iter = iter + 1u) {
      let blk = philox4x32_10(philox_counter_offset(base_c, idx, iter), key);
      let z_pair = philox_box_muller(blk.x, blk.y);
      if (z_pair.x >= low_z && z_pair.x <= high_z) {
        chosen_z = z_pair.x;
        accepted = true;
        break;
      }
      if (z_pair.y >= low_z && z_pair.y <= high_z) {
        chosen_z = z_pair.y;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      let blk = philox4x32_10(philox_counter_offset(base_c, idx, 64u), key);
      let u = philox_u01_open_f32(blk.x);
      chosen_z = clamp(low_z + u * (high_z - low_z), low_z, high_z);
    }
    dst_buf[phys_idx] = f64_add(p1, f64_mul(p2, f64_from_f32(chosen_z)));
  } else if (params.mode == 5u) {
    // gamma(alpha, scale)
    let alpha = bitcast<f32>(params.param0_lo);
    let g = sample_gamma_unit(alpha, base_c, key, idx, 0u);
    dst_buf[phys_idx] = f64_mul(p1, f64_from_f32(g));
  } else if (params.mode == 6u) {
    // beta(a, b) = X / (X + Y)
    let a_param = bitcast<f32>(params.param0_lo);
    let b_param = bitcast<f32>(params.param0_hi);
    let x_g = sample_gamma_unit(a_param, base_c, key, idx, 0u);
    let y_g = sample_gamma_unit(b_param, base_c, key, idx, 128u);
    let ratio = clamp(x_g / (x_g + y_g), 0.0, 1.0);
    dst_buf[phys_idx] = f64_from_f32(ratio);
  }
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'random_float64',
  );
}

/// Builds a WGSL compute shader for discrete `Int64` random distributions:
/// - `mode == 0`: `randint(low, high)` (`param0 = low_i64`, `param1 = span_u64`)
/// - `mode == 1`: `poisson(lam)` (`param0_lo = lam_f32`)
/// - `mode == 2`: `binomial(n, p)` (`param0_lo = n_u32`, `param0_hi = p_f32`)
WgslShaderModule buildRandomInt64Shader() {
  const code =
      '''
$wgslPhilox4x32Lib
$_wgslOutLayoutStructAndHelpers

@group(0) @binding(0) var<storage, read_write> dst_buf: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> params: RngUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total_elements) {
    return;
  }
  let base_c = vec4<u32>(params.counter0, params.counter1, params.counter2, params.counter3);
  let key = vec2<u32>(params.key0, params.key1);
  let phys_idx = compute_out_phys_idx(params, idx);

  if (params.mode == 0u) {
    // randint: low + (rand_u64 % span) with unbiased rejection for 32-bit spans
    let low_i64 = vec2<u32>(params.param0_lo, params.param0_hi);
    let span_lo = params.param1_lo;
    let span_hi = params.param1_hi;
    var offset_u64 = vec2<u32>(0u, 0u);
    if (span_hi == 0u) {
      if (span_lo > 1u) {
        let threshold = (0u - span_lo) % span_lo;
        for (var iter = 0u; iter < 16u; iter = iter + 1u) {
          let blk = philox4x32_10(philox_counter_offset(base_c, idx, iter), key);
          if (blk.x >= threshold) {
            offset_u64 = vec2<u32>(blk.x % span_lo, 0u);
            break;
          }
        }
      }
    } else {
      let blk = philox4x32_10(philox_counter_offset(base_c, idx, 0u), key);
      offset_u64 = vec2<u32>(blk.x, blk.y % span_hi);
    }
    dst_buf[phys_idx] = u64_add(low_i64, offset_u64);
  } else if (params.mode == 1u) {
    // poisson(lam)
    let lam = bitcast<f32>(params.param0_lo);
    var k_count = 0u;
    if (lam < 30.0) {
      let target_l = exp(-lam);
      var prod = 1.0;
      for (var iter = 0u; iter < 256u; iter = iter + 1u) {
        let blk = philox4x32_10(philox_counter_offset(base_c, idx, iter), key);
        prod = prod * philox_u01_open_f32(blk.x);
        if (prod <= target_l) {
          k_count = iter;
          break;
        }
      }
    } else {
      let blk = philox4x32_10(philox_counter_offset(base_c, idx, 0u), key);
      let z = philox_box_muller(blk.x, blk.y).x;
      let approx = floor(lam + sqrt(lam) * z + 0.5);
      k_count = u32(max(0.0, approx));
    }
    dst_buf[phys_idx] = vec2<u32>(k_count, 0u);
  } else if (params.mode == 2u) {
    // binomial(n, p)
    let trials = params.param0_lo;
    let prob = bitcast<f32>(params.param0_hi);
    var successes = 0u;
    if (prob <= 0.0 || trials == 0u) {
      successes = 0u;
    } else if (prob >= 1.0) {
      successes = trials;
    } else if (trials <= 128u) {
      for (var t = 0u; t < trials; t = t + 4u) {
        let blk = philox4x32_10(philox_counter_offset(base_c, idx, t >> 2u), key);
        if (t < trials && philox_u01_open_f32(blk.x) < prob) { successes = successes + 1u; }
        if (t + 1u < trials && philox_u01_open_f32(blk.y) < prob) { successes = successes + 1u; }
        if (t + 2u < trials && philox_u01_open_f32(blk.z) < prob) { successes = successes + 1u; }
        if (t + 3u < trials && philox_u01_open_f32(blk.w) < prob) { successes = successes + 1u; }
      }
    } else {
      let blk = philox4x32_10(philox_counter_offset(base_c, idx, 0u), key);
      let z = philox_box_muller(blk.x, blk.y).x;
      let mean = f32(trials) * prob;
      let std_dev = sqrt(mean * (1.0 - prob));
      let approx = clamp(floor(mean + std_dev * z + 0.5), 0.0, f32(trials));
      successes = u32(approx);
    }
    dst_buf[phys_idx] = vec2<u32>(successes, 0u);
  }
}
''';
  return WgslShaderModule(code: code, entryPoint: 'main', name: 'random_int64');
}
