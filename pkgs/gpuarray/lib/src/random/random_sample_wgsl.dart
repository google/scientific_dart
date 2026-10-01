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
import 'random_wgsl.dart';

/// Builds a WGSL shader that generates a bijective random permutation of
/// `[0, 1, ..., n - 1]` as `Int64` (`vec2<u32>`) using a cycle-walking
/// generalized Feistel network keyed by `Philox4x32-10`.
WgslShaderModule buildRandomPermutationInt64Shader() {
  const code =
      '''
$wgslPhilox4x32Lib

struct PermUniforms {
  n: u32,
  bits: u32,
  key0: u32,
  key1: u32,
  counter0: u32,
  counter1: u32,
  counter2: u32,
  counter3: u32,
}

@group(0) @binding(0) var<storage, read_write> dst_buf: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> params: PermUniforms;

fn feistel_prf(val: u32, round_idx: u32, base_c: vec4<u32>, key: vec2<u32>) -> u32 {
  let ctr = vec4<u32>(base_c.x ^ val, base_c.y + round_idx, base_c.z, base_c.w);
  let blk = philox4x32_10(ctr, key);
  return blk.x ^ blk.z;
}

fn feistel_bijection(x_in: u32, bits: u32, base_c: vec4<u32>, key: vec2<u32>) -> u32 {
  if (bits == 0u) {
    return 0u;
  }
  let half_l = bits >> 1u;
  let half_r = bits - half_l;
  let mask_l = select(0u, (1u << half_l) - 1u, half_l > 0u);
  let mask_r = (1u << half_r) - 1u;

  var left = (x_in >> half_r) & mask_l;
  var right = x_in & mask_r;

  for (var r = 0u; r < 4u; r = r + 1u) {
    if (half_l > 0u) {
      left = (left ^ feistel_prf(right, r * 2u, base_c, key)) & mask_l;
    }
    right = (right ^ feistel_prf(left, r * 2u + 1u, base_c, key)) & mask_r;
  }
  return (left << half_r) | right;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.n) {
    return;
  }
  if (params.n == 1u) {
    dst_buf[0] = vec2<u32>(0u, 0u);
    return;
  }
  let base_c = vec4<u32>(params.counter0, params.counter1, params.counter2, params.counter3);
  let key = vec2<u32>(params.key0, params.key1);

  var curr = idx;
  for (var step = 0u; step <= params.n; step = step + 1u) {
    curr = feistel_bijection(curr, params.bits, base_c, key);
    if (curr < params.n) {
      break;
    }
  }
  dst_buf[idx] = vec2<u32>(curr, 0u);
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'random_permutation_int64',
  );
}

String _elementCopyDeclarationAndBody(DType dtype) {
  final byteWidth = dtype.byteWidth;
  if (byteWidth == 16) {
    return '''
@group(0) @binding(0) var<storage, read> src_buf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> idx_buf: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<vec4<u32>>;
fn copy_elem(src_phys: u32, dst_phys: u32) {
  dst_buf[dst_phys] = src_buf[src_phys];
}
''';
  }
  if (byteWidth == 8) {
    return '''
@group(0) @binding(0) var<storage, read> src_buf: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> idx_buf: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<vec2<u32>>;
fn copy_elem(src_phys: u32, dst_phys: u32) {
  dst_buf[dst_phys] = src_buf[src_phys];
}
''';
  }
  if (byteWidth == 4) {
    return '''
@group(0) @binding(0) var<storage, read> src_buf: array<u32>;
@group(0) @binding(1) var<storage, read> idx_buf: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<u32>;
fn copy_elem(src_phys: u32, dst_phys: u32) {
  dst_buf[dst_phys] = src_buf[src_phys];
}
''';
  }
  if (byteWidth == 2) {
    return '''
@group(0) @binding(0) var<storage, read_write> src_buf: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> idx_buf: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<atomic<u32>>;
fn copy_elem(src_phys: u32, dst_phys: u32) {
  let src_word = atomicLoad(&src_buf[src_phys >> 1u]);
  let val16 = (src_word >> ((src_phys & 1u) * 16u)) & 0xFFFFu;
  let dst_shift = (dst_phys & 1u) * 16u;
  let mask = ~(0xFFFFu << dst_shift);
  atomicAnd(&dst_buf[dst_phys >> 1u], mask);
  atomicOr(&dst_buf[dst_phys >> 1u], val16 << dst_shift);
}
''';
  }
  return '''
@group(0) @binding(0) var<storage, read_write> src_buf: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> idx_buf: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<atomic<u32>>;
fn copy_elem(src_phys: u32, dst_phys: u32) {
  let src_word = atomicLoad(&src_buf[src_phys >> 2u]);
  let val8 = (src_word >> ((src_phys & 3u) * 8u)) & 0xFFu;
  let dst_shift = (dst_phys & 3u) * 8u;
  let mask = ~(0xFFu << dst_shift);
  atomicAnd(&dst_buf[dst_phys >> 2u], mask);
  atomicOr(&dst_buf[dst_phys >> 2u], val8 << dst_shift);
}
''';
}

/// Builds a WGSL shader that gathers slices along axis 0 of an N-D array of [dtype]
/// according to an `Int64` permutation buffer `idx_buf`.
WgslShaderModule buildRandomAxis0SliceGatherShader(DType dtype) {
  final copyDecl = _elementCopyDeclarationAndBody(dtype);
  final code =
      '''
struct SliceGatherUniforms {
  total_elements: u32,
  slice_size: u32,
  rank: u32,
  src_offset: u32,
  dst_offset: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  shape0: vec4<u32>,
  shape1: vec4<u32>,
  src_strides0: vec4<i32>,
  src_strides1: vec4<i32>,
  dst_strides0: vec4<i32>,
  dst_strides1: vec4<i32>,
}

$copyDecl
@group(0) @binding(3) var<uniform> params: SliceGatherUniforms;

fn get_dim(d: u32) -> u32 {
  if (d < 4u) { return params.shape0[d]; }
  return params.shape1[d - 4u];
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
  let out_row = linear_idx / params.slice_size;
  let src_row = idx_buf[out_row].x;

  var rem = linear_idx % params.slice_size;
  var src_off = i32(params.src_offset) + i32(src_row) * get_src_stride(0u);
  var dst_off = i32(params.dst_offset) + i32(out_row) * get_dst_stride(0u);

  for (var i = 1u; i < params.rank; i = i + 1u) {
    let d = params.rank - i;
    let dim_size = get_dim(d);
    let coord = rem % dim_size;
    rem = rem / dim_size;
    src_off = src_off + i32(coord) * get_src_stride(d);
    dst_off = dst_off + i32(coord) * get_dst_stride(d);
  }
  copy_elem(u32(src_off), u32(dst_off));
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'random_slice_gather_${dtype.name}',
  );
}

/// Builds a WGSL shader that gathers elements from a 1D array `src_buf` of [dtype]
/// using an `Int64` index buffer `idx_buf` into an arbitrary-rank destination `dst_buf`.
WgslShaderModule buildRandomGather1dShader(DType dtype) {
  final copyDecl = _elementCopyDeclarationAndBody(dtype);
  final code =
      '''
struct Gather1dUniforms {
  total_elements: u32,
  out_rank: u32,
  src_offset: u32,
  src_stride: i32,
  dst_offset: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  out_shape0: vec4<u32>,
  out_shape1: vec4<u32>,
  out_strides0: vec4<i32>,
  out_strides1: vec4<i32>,
}

$copyDecl
@group(0) @binding(3) var<uniform> params: Gather1dUniforms;

fn get_out_dim(d: u32) -> u32 {
  if (d < 4u) { return params.out_shape0[d]; }
  return params.out_shape1[d - 4u];
}

fn get_out_stride(d: u32) -> i32 {
  if (d < 4u) { return params.out_strides0[d]; }
  return params.out_strides1[d - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear_idx = gid.x;
  if (linear_idx >= params.total_elements) {
    return;
  }
  let chosen_idx = idx_buf[linear_idx].x;
  let src_phys = u32(i32(params.src_offset) + i32(chosen_idx) * params.src_stride);

  var dst_off = i32(params.dst_offset);
  if (params.out_rank > 0u) {
    var rem = linear_idx;
    for (var i = 0u; i < params.out_rank; i = i + 1u) {
      let d = params.out_rank - 1u - i;
      let dim_size = get_out_dim(d);
      let coord = rem % dim_size;
      rem = rem / dim_size;
      dst_off = dst_off + i32(coord) * get_out_stride(d);
    }
  }
  copy_elem(src_phys, u32(dst_off));
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'random_gather1d_${dtype.name}',
  );
}

/// Builds a WGSL shader that samples `Int64` indices from a 1D probability
/// vector `prob_buf` (`f32`) with or without replacement on the GPU.
WgslShaderModule buildRandomWeightedIndexShader() {
  const code =
      '''
$wgslPhilox4x32Lib

struct WeightedUniforms {
  sample_count: u32,
  pop_size: u32,
  replace_flag: u32,
  pad0: u32,
  key0: u32,
  key1: u32,
  counter0: u32,
  counter1: u32,
  counter2: u32,
  counter3: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> prob_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> idx_buf: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: WeightedUniforms;

fn exp_race_key(item_idx: u32, base_c: vec4<u32>, key: vec2<u32>) -> f32 {
  let w = prob_buf[item_idx];
  if (w <= 0.0) {
    return 1e30;
  }
  let blk = philox4x32_10(philox_counter_offset(base_c, item_idx, 0u), key);
  let u = philox_u01_open_f32(blk.x);
  return -log(u) / w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let base_c = vec4<u32>(params.counter0, params.counter1, params.counter2, params.counter3);
  let key = vec2<u32>(params.key0, params.key1);

  if (params.replace_flag != 0u) {
    let sample_idx = gid.x;
    if (sample_idx >= params.sample_count) {
      return;
    }
    let blk = philox4x32_10(philox_counter_offset(base_c, sample_idx, 0u), key);
    let u = philox_u01_open_f32(blk.x);
    var cum = 0.0;
    var chosen = params.pop_size - 1u;
    for (var k = 0u; k < params.pop_size; k = k + 1u) {
      cum = cum + prob_buf[k];
      if (u < cum) {
        chosen = k;
        break;
      }
    }
    idx_buf[sample_idx] = vec2<u32>(chosen, 0u);
  } else {
    // Weighted sampling without replacement via Efraimidis-Spirakis exponential race keys.
    let item_idx = gid.x;
    if (item_idx >= params.pop_size) {
      return;
    }
    let my_key = exp_race_key(item_idx, base_c, key);
    var rank = 0u;
    for (var j = 0u; j < params.pop_size; j = j + 1u) {
      if (j == item_idx) {
        continue;
      }
      let other_key = exp_race_key(j, base_c, key);
      if (other_key < my_key || (other_key == my_key && j < item_idx)) {
        rank = rank + 1u;
      }
    }
    if (rank < params.sample_count) {
      idx_buf[rank] = vec2<u32>(item_idx, 0u);
    }
  }
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'random_weighted_index',
  );
}

/// Builds a WGSL shader for `categorical` sampling from logits or probabilities
/// of [sourceDType] using the Gumbel-max trick on the GPU.
WgslShaderModule buildRandomCategoricalShader(DType sourceDType) {
  final String storageType;
  final String loadF32Expr;
  switch (sourceDType) {
    case DType.float64:
      storageType = 'vec2<u32>';
      loadF32Expr = '''
  let bits64 = logits_buf[phys_logit];
  let sign = bits64.y & 0x80000000u;
  let exp64 = i32((bits64.y >> 20u) & 0x7FFu);
  var logit_val = 0.0;
  if (exp64 > 0) {
    let exp32 = clamp(exp64 - 1023 + 127, 1, 254);
    let mant23 = ((bits64.y & 0x000FFFFFu) << 3u) | (bits64.x >> 29u);
    logit_val = bitcast<f32>(sign | (u32(exp32) << 23u) | mant23);
  }
''';
    case DType.float32:
      storageType = 'f32';
      loadF32Expr = 'let logit_val = logits_buf[phys_logit];';
    default:
      storageType = 'f32';
      loadF32Expr = 'let logit_val = logits_buf[phys_logit];';
  }
  final code =
      '''
$wgslPhilox4x32Lib

struct CategoricalUniforms {
  sample_count: u32,
  batch_rows: u32,
  num_classes: u32,
  from_logits: u32,
  logits_offset: u32,
  row_stride: i32,
  class_stride: i32,
  out_rank: u32,
  out_offset: u32,
  key0: u32,
  key1: u32,
  counter0: u32,
  counter1: u32,
  counter2: u32,
  counter3: u32,
  pad0: u32,
  out_shape0: vec4<u32>,
  out_shape1: vec4<u32>,
  out_strides0: vec4<i32>,
  out_strides1: vec4<i32>,
}

@group(0) @binding(0) var<storage, read> logits_buf: array<$storageType>;
@group(0) @binding(1) var<storage, read_write> dst_buf: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: CategoricalUniforms;

fn get_out_dim(d: u32) -> u32 {
  if (d < 4u) { return params.out_shape0[d]; }
  return params.out_shape1[d - 4u];
}

fn get_out_stride(d: u32) -> i32 {
  if (d < 4u) { return params.out_strides0[d]; }
  return params.out_strides1[d - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let sample_idx = gid.x;
  if (sample_idx >= params.sample_count) {
    return;
  }
  let base_c = vec4<u32>(params.counter0, params.counter1, params.counter2, params.counter3);
  let key = vec2<u32>(params.key0, params.key1);
  let row_idx = sample_idx % params.batch_rows;
  let row_base = i32(params.logits_offset) + i32(row_idx) * params.row_stride;

  var best_class = 0u;
  var best_score = -1e38;
  for (var c = 0u; c < params.num_classes; c = c + 1u) {
    let phys_logit = u32(row_base + i32(c) * params.class_stride);
    $loadF32Expr
    let score_base = select(log(max(logit_val, 1e-30)), logit_val, params.from_logits != 0u);
    let blk = philox4x32_10(philox_counter_offset(base_c, sample_idx, c), key);
    let u = philox_u01_open_f32(blk.x);
    let gumbel = -log(-log(u));
    let perturbed = score_base + gumbel;
    if (c == 0u || perturbed > best_score) {
      best_score = perturbed;
      best_class = c;
    }
  }

  var dst_off = i32(params.out_offset);
  if (params.out_rank > 0u) {
    var rem = sample_idx;
    for (var i = 0u; i < params.out_rank; i = i + 1u) {
      let d = params.out_rank - 1u - i;
      let dim_size = get_out_dim(d);
      let coord = rem % dim_size;
      rem = rem / dim_size;
      dst_off = dst_off + i32(coord) * get_out_stride(d);
    }
  }
  dst_buf[u32(dst_off)] = vec2<u32>(best_class, 0u);
}
''';
  return WgslShaderModule(
    code: code,
    entryPoint: 'main',
    name: 'random_categorical_${sourceDType.name}',
  );
}
