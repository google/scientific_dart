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
import 'fft_wgsl.dart';

/// Builds a WGSL shader for Bluestein's Chirp-Z pre-multiplication and padding
/// into length `chirpM` (power of 2 >= `2 * n - 1`).
WgslShaderModule buildFftBluesteinPreShader() {
  const code =
      '''
$wgslDoubleFloatComplexLib

struct BluesteinPreUniforms {
  batch_count: u32,
  n: u32,
  chirp_m: u32,
  sign_bits: u32,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> a_pad_buf: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> b_pad_buf: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> params: BluesteinPreUniforms;

fn bluestein_chirp(idx: u32, n: u32, sign_dir: f32) -> vec4<u32> {
  let two_n = 2u * n;
  let k_mod = idx % two_n;
  let sq_mod = (k_mod * k_mod) % two_n;
  return cdf_twiddle_ratio(sq_mod, two_n, sign_dir);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.chirp_m;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.chirp_m;
  let k = linear_idx % params.chirp_m;
  let sign_dir = bitcast<f32>(params.sign_bits);

  if (k < params.n) {
    let w = bluestein_chirp(k, params.n, sign_dir);
    let x_val = unpack_c128_cdf(src_buf[batch_idx * params.n + k]);
    a_pad_buf[linear_idx] = pack_cdf_c128(cdf_mul(x_val, w));
    b_pad_buf[linear_idx] = pack_cdf_c128(cdf_conj(w));
  } else if (k > params.chirp_m - params.n) {
    a_pad_buf[linear_idx] = vec4<u32>(0u, 0u, 0u, 0u);
    let mirror = params.chirp_m - k;
    let w = bluestein_chirp(mirror, params.n, sign_dir);
    b_pad_buf[linear_idx] = pack_cdf_c128(cdf_conj(w));
  } else {
    a_pad_buf[linear_idx] = vec4<u32>(0u, 0u, 0u, 0u);
    b_pad_buf[linear_idx] = vec4<u32>(0u, 0u, 0u, 0u);
  }
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_bluestein_pre',
  );
}

/// Builds a WGSL shader for pointwise complex double-float multiplication.
WgslShaderModule buildFftComplexMulShader() {
  const code =
      '''
$wgslDoubleFloatComplexLib

struct MulUniforms {
  total_elements: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> a_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> b_buf: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> params: MulUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total_elements) {
    return;
  }
  let a_val = unpack_c128_cdf(a_buf[idx]);
  let b_val = unpack_c128_cdf(b_buf[idx]);
  dst_buf[idx] = pack_cdf_c128(cdf_mul(a_val, b_val));
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_complex_mul',
  );
}

/// Builds a WGSL shader for Bluestein's Chirp-Z post-multiplication and extraction
/// of the first `n` bins per batch row.
WgslShaderModule buildFftBluesteinPostShader() {
  const code =
      '''
$wgslDoubleFloatComplexLib

struct BluesteinPostUniforms {
  batch_count: u32,
  n: u32,
  chirp_m: u32,
  sign_bits: u32,
  inv_m_hi: u32,
  inv_m_lo: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> conv_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: BluesteinPostUniforms;

fn bluestein_chirp(idx: u32, n: u32, sign_dir: f32) -> vec4<u32> {
  let two_n = 2u * n;
  let k_mod = idx % two_n;
  let sq_mod = (k_mod * k_mod) % two_n;
  return cdf_twiddle_ratio(sq_mod, two_n, sign_dir);
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
  let sign_dir = bitcast<f32>(params.sign_bits);
  let inv_m = vec2<u32>(params.inv_m_hi, params.inv_m_lo);
  let w = bluestein_chirp(k, params.n, sign_dir);
  let raw = unpack_c128_cdf(conv_buf[batch_idx * params.chirp_m + k]);
  let scaled = cdf_scale(cdf_mul(raw, w), inv_m);
  dst_buf[linear_idx] = pack_cdf_c128(scaled);
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_bluestein_post',
  );
}

/// Builds a WGSL shader that scatters `[batchCount, outAxisLength]` from a
/// `[batchCount, srcAxisLength]` `Complex128` buffer into a potentially strided
/// N-D `Complex128` destination buffer, applying normalization scaling and optional conjugation.
WgslShaderModule buildFftScatterComplexShader() {
  const code =
      '''
$wgslDoubleFloatComplexLib

struct ScatterComplexUniforms {
  batch_count: u32,
  src_axis_length: u32,
  out_axis_length: u32,
  outer_rank: u32,
  out_offset: u32,
  out_axis_stride: i32,
  scale_hi: u32,
  scale_lo: u32,
  conjugate_output: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  outer_shape0: vec4<u32>,
  outer_shape1: vec4<u32>,
  out_outer_strides0: vec4<i32>,
  out_outer_strides1: vec4<i32>,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: ScatterComplexUniforms;

fn get_outer_dim(d: u32) -> u32 {
  if (d < 4u) { return params.outer_shape0[d]; }
  return params.outer_shape1[d - 4u];
}

fn get_out_outer_stride(d: u32) -> i32 {
  if (d < 4u) { return params.out_outer_strides0[d]; }
  return params.out_outer_strides1[d - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch_count * params.out_axis_length;
  let linear_idx = gid.x;
  if (linear_idx >= total) {
    return;
  }
  let batch_idx = linear_idx / params.out_axis_length;
  let k = linear_idx % params.out_axis_length;

  var rem = batch_idx;
  var base_offset = i32(params.out_offset);
  for (var i = 0u; i < params.outer_rank; i = i + 1u) {
    let d = params.outer_rank - 1u - i;
    let dim_size = get_outer_dim(d);
    let coord = rem % dim_size;
    rem = rem / dim_size;
    base_offset = base_offset + i32(coord) * get_out_outer_stride(d);
  }
  let phys_idx = u32(base_offset + i32(k) * params.out_axis_stride);

  let scale = vec2<u32>(params.scale_hi, params.scale_lo);
  var val = cdf_scale(unpack_c128_cdf(src_buf[batch_idx * params.src_axis_length + k]), scale);
  if (params.conjugate_output != 0u) {
    val = cdf_conj(val);
  }
  dst_buf[phys_idx] = pack_cdf_c128(val);
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_scatter_complex',
  );
}

/// Builds a WGSL shader that extracts and scales the real part of `[batchCount, n]`
/// from a `Complex128` buffer into a potentially strided N-D `Float64` destination buffer.
WgslShaderModule buildFftScatterRealShader() {
  const code =
      '''
$wgslDoubleFloatComplexLib

struct ScatterRealUniforms {
  batch_count: u32,
  n: u32,
  outer_rank: u32,
  out_offset: u32,
  out_axis_stride: i32,
  scale_hi: u32,
  scale_lo: u32,
  pad0: u32,
  outer_shape0: vec4<u32>,
  outer_shape1: vec4<u32>,
  out_outer_strides0: vec4<i32>,
  out_outer_strides1: vec4<i32>,
}

@group(0) @binding(0) var<storage, read> src_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: ScatterRealUniforms;

fn get_outer_dim(d: u32) -> u32 {
  if (d < 4u) { return params.outer_shape0[d]; }
  return params.outer_shape1[d - 4u];
}

fn get_out_outer_stride(d: u32) -> i32 {
  if (d < 4u) { return params.out_outer_strides0[d]; }
  return params.out_outer_strides1[d - 4u];
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

  var rem = batch_idx;
  var base_offset = i32(params.out_offset);
  for (var i = 0u; i < params.outer_rank; i = i + 1u) {
    let d = params.outer_rank - 1u - i;
    let dim_size = get_outer_dim(d);
    let coord = rem % dim_size;
    rem = rem / dim_size;
    base_offset = base_offset + i32(coord) * get_out_outer_stride(d);
  }
  let phys_idx = u32(base_offset + i32(k) * params.out_axis_stride);

  let scale = vec2<u32>(params.scale_hi, params.scale_lo);
  let re_df = unpack_f64_df(src_buf[batch_idx * params.n + k].xy);
  let scaled_re = df_mul(re_df, scale);
  dst_buf[phys_idx] = pack_df_f64(scaled_re);
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_scatter_real',
  );
}

/// Builds a WGSL shader that generates `fftfreq` or `rfftfreq` values directly
/// on the GPU in 48-bit double-float precision and writes `Float64` (`vec2<u32>`).
WgslShaderModule buildFftFreqShader() {
  const code =
      '''
$wgslDoubleFloatComplexLib

struct FreqUniforms {
  out_length: u32,
  n: u32,
  positive_count: u32,
  is_rfft: u32,
  out_offset: u32,
  out_stride: i32,
  denom_hi: u32,
  denom_lo: u32,
}

@group(0) @binding(0) var<storage, read_write> dst_buf: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> params: FreqUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.out_length) {
    return;
  }
  var signed_k = i32(k);
  if (params.is_rfft == 0u && k >= params.positive_count) {
    signed_k = i32(k) - i32(params.n);
  }
  let denom = vec2<u32>(params.denom_hi, params.denom_lo);
  let freq = df_div(f64_from_i32(signed_k), denom);
  let phys_idx = u32(i32(params.out_offset) + i32(k) * params.out_stride);
  dst_buf[phys_idx] = pack_df_f64(freq);
}
''';
  return WgslShaderModule(code: code, entryPoint: 'main', name: 'fft_freq');
}

/// Builds a WGSL shader that performs N-D `fftshift` or `ifftshift` across arbitrary
/// axes and strided views for any element [dtype] directly on the GPU.
WgslShaderModule buildFftShiftShader(DType dtype) {
  final bytesPerElement = dtype.byteWidth;
  final String storageType;
  final String copyStmt;
  if (bytesPerElement == 16) {
    storageType = 'vec4<u32>';
    copyStmt = 'dst_buf[dst_phys] = src_buf[src_phys];';
  } else if (bytesPerElement == 8) {
    storageType = 'vec2<u32>';
    copyStmt = 'dst_buf[dst_phys] = src_buf[src_phys];';
  } else if (bytesPerElement == 4) {
    storageType = 'u32';
    copyStmt = 'dst_buf[dst_phys] = src_buf[src_phys];';
  } else if (bytesPerElement == 2) {
    storageType = 'atomic<u32>';
    copyStmt = '''
  let src_word = atomicLoad(&src_buf[src_phys >> 1u]);
  let val16 = (src_word >> ((src_phys & 1u) * 16u)) & 0xFFFFu;
  let dst_shift = (dst_phys & 1u) * 16u;
  let mask = ~(0xFFFFu << dst_shift);
  atomicAnd(&dst_buf[dst_phys >> 1u], mask);
  atomicOr(&dst_buf[dst_phys >> 1u], val16 << dst_shift);
''';
  } else {
    storageType = 'atomic<u32>';
    copyStmt = '''
  let src_word = atomicLoad(&src_buf[src_phys >> 2u]);
  let val8 = (src_word >> ((src_phys & 3u) * 8u)) & 0xFFu;
  let dst_shift = (dst_phys & 3u) * 8u;
  let mask = ~(0xFFu << dst_shift);
  atomicAnd(&dst_buf[dst_phys >> 2u], mask);
  atomicOr(&dst_buf[dst_phys >> 2u], val8 << dst_shift);
''';
  }
  final srcAccess = bytesPerElement < 4 ? 'read_write' : 'read';
  final code =
      '''
struct ShiftUniforms {
  total_elements: u32,
  rank: u32,
  src_offset: u32,
  dst_offset: u32,
  shape0: vec4<u32>,
  shape1: vec4<u32>,
  shifts0: vec4<u32>,
  shifts1: vec4<u32>,
  src_strides0: vec4<i32>,
  src_strides1: vec4<i32>,
  dst_strides0: vec4<i32>,
  dst_strides1: vec4<i32>,
}

@group(0) @binding(0) var<storage, $srcAccess> src_buf: array<$storageType>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<$storageType>;
@group(0) @binding(2) var<uniform> params: ShiftUniforms;

fn get_dim(d: u32) -> u32 {
  if (d < 4u) { return params.shape0[d]; }
  return params.shape1[d - 4u];
}

fn get_shift(d: u32) -> u32 {
  if (d < 4u) { return params.shifts0[d]; }
  return params.shifts1[d - 4u];
}

fn get_src_stride(d: u32) -> i32 {
  if (d < 4u) { return params.src_strides0[d]; }
  return params.src_strides1[d - 4u];
}

fn get_dst_stride(d: u32) -> i32 {
  if (d < 4u) { return params.dst_strides0[d]; }
  return params.dst_strides1[d - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear_idx = gid.x;
  if (linear_idx >= params.total_elements) {
    return;
  }
  var rem = linear_idx;
  var src_off = i32(params.src_offset);
  var dst_off = i32(params.dst_offset);
  for (var i = 0u; i < params.rank; i = i + 1u) {
    let d = params.rank - 1u - i;
    let dim_size = get_dim(d);
    let dst_coord = rem % dim_size;
    rem = rem / dim_size;
    let shift_val = get_shift(d);
    let src_coord = (dst_coord + dim_size - (shift_val % dim_size)) % dim_size;
    src_off = src_off + i32(src_coord) * get_src_stride(d);
    dst_off = dst_off + i32(dst_coord) * get_dst_stride(d);
  }
  let src_phys = u32(src_off);
  let dst_phys = u32(dst_off);
  $copyStmt
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'fft_shift_${dtype.name}',
  );
}
