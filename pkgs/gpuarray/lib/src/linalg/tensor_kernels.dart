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

import 'dart:math' as math;

import '../buffer.dart';
import '../device.dart';
import 'linalg_wgsl_df64.dart';

const String _batchedMatmulRealShader =
    '''
$linalgDf64WgslLibrary

struct MatmulParams {
  batch_count: u32,
  m: u32,
  k: u32,
  n: u32,
  batch_ndim: u32,
  transpose_b: u32,
  pad0: u32,
  pad1: u32,
  batch_shape0: vec4<u32>,
  batch_shape1: vec4<u32>,
  a_batch_shape0: vec4<u32>,
  a_batch_shape1: vec4<u32>,
  b_batch_shape0: vec4<u32>,
  b_batch_shape1: vec4<u32>,
};

fn get_vec8(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
  switch (d) {
    case 0u: { return v0.x; }
    case 1u: { return v0.y; }
    case 2u: { return v0.z; }
    case 3u: { return v0.w; }
    case 4u: { return v1.x; }
    case 5u: { return v1.y; }
    case 6u: { return v1.z; }
    default: { return v1.w; }
  }
}

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_c: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let mn = params.m * params.n;
  let total = params.batch_count * mn;
  if (idx >= total) { return; }

  let batch_idx = idx / mn;
  let rem_mn = idx % mn;
  let row = rem_mn / params.n;
  let col = rem_mn % params.n;

  var a_batch = 0u;
  var b_batch = 0u;
  var a_stride = 1u;
  var b_stride = 1u;
  var b_rem = batch_idx;
  if (params.batch_ndim > 0u) {
    var d = i32(params.batch_ndim) - 1;
    loop {
      if (d < 0) { break; }
      let ud = u32(d);
      let b_sz = max(1u, get_vec8(params.batch_shape0, params.batch_shape1, ud));
      let a_sz = max(1u, get_vec8(params.a_batch_shape0, params.a_batch_shape1, ud));
      let bb_sz = max(1u, get_vec8(params.b_batch_shape0, params.b_batch_shape1, ud));
      let coord = b_rem % b_sz;
      b_rem = b_rem / b_sz;
      a_batch = a_batch + (coord % a_sz) * a_stride;
      b_batch = b_batch + (coord % bb_sz) * b_stride;
      a_stride = a_stride * a_sz;
      b_stride = b_stride * bb_sz;
      d = d - 1;
    }
  }

  let a_base = (a_batch * params.m + row) * params.k;
  let b_mat_base = b_batch * params.k * params.n;
  var acc = df64_zero();
  for (var p = 0u; p < params.k; p = p + 1u) {
    let av = unpack_f64_df64(in_a[a_base + p]);
    var b_idx = b_mat_base + p * params.n + col;
    if (params.transpose_b != 0u) {
      b_idx = b_mat_base + col * params.k + p;
    }
    let bv = unpack_f64_df64(in_b[b_idx]);
    acc = df64_add(acc, df64_mul(av, bv));
  }
  out_c[idx] = pack_df64_f64(acc);
}
''';

const String _batchedMatmulComplexShader =
    '''
$linalgDf64WgslLibrary

struct MatmulParams {
  batch_count: u32,
  m: u32,
  k: u32,
  n: u32,
  batch_ndim: u32,
  transpose_b: u32,
  conjugate_a: u32,
  pad1: u32,
  batch_shape0: vec4<u32>,
  batch_shape1: vec4<u32>,
  a_batch_shape0: vec4<u32>,
  a_batch_shape1: vec4<u32>,
  b_batch_shape0: vec4<u32>,
  b_batch_shape1: vec4<u32>,
};

fn get_vec8(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
  switch (d) {
    case 0u: { return v0.x; }
    case 1u: { return v0.y; }
    case 2u: { return v0.z; }
    case 3u: { return v0.w; }
    case 4u: { return v1.x; }
    case 5u: { return v1.y; }
    case 6u: { return v1.z; }
    default: { return v1.w; }
  }
}

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_c: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let mn = params.m * params.n;
  let total = params.batch_count * mn;
  if (idx >= total) { return; }

  let batch_idx = idx / mn;
  let rem_mn = idx % mn;
  let row = rem_mn / params.n;
  let col = rem_mn % params.n;

  var a_batch = 0u;
  var b_batch = 0u;
  var a_stride = 1u;
  var b_stride = 1u;
  var b_rem = batch_idx;
  if (params.batch_ndim > 0u) {
    var d = i32(params.batch_ndim) - 1;
    loop {
      if (d < 0) { break; }
      let ud = u32(d);
      let b_sz = max(1u, get_vec8(params.batch_shape0, params.batch_shape1, ud));
      let a_sz = max(1u, get_vec8(params.a_batch_shape0, params.a_batch_shape1, ud));
      let bb_sz = max(1u, get_vec8(params.b_batch_shape0, params.b_batch_shape1, ud));
      let coord = b_rem % b_sz;
      b_rem = b_rem / b_sz;
      a_batch = a_batch + (coord % a_sz) * a_stride;
      b_batch = b_batch + (coord % bb_sz) * b_stride;
      a_stride = a_stride * a_sz;
      b_stride = b_stride * bb_sz;
      d = d - 1;
    }
  }

  let a_base = (a_batch * params.m + row) * params.k;
  let b_mat_base = b_batch * params.k * params.n;
  var acc = cdf64_zero();
  for (var p = 0u; p < params.k; p = p + 1u) {
    let a_pos = 2u * (a_base + p);
    let a_re = unpack_f64_df64(in_a[a_pos]);
    let a_im = unpack_f64_df64(in_a[a_pos + 1u]);
    var av = vec4<f32>(a_re.x, a_re.y, a_im.x, a_im.y);
    if (params.conjugate_a != 0u) {
      av = cdf64_conj(av);
    }
    var b_elem = b_mat_base + p * params.n + col;
    if (params.transpose_b != 0u) {
      b_elem = b_mat_base + col * params.k + p;
    }
    let b_pos = 2u * b_elem;
    let b_re = unpack_f64_df64(in_b[b_pos]);
    let b_im = unpack_f64_df64(in_b[b_pos + 1u]);
    let bv = vec4<f32>(b_re.x, b_re.y, b_im.x, b_im.y);
    acc = cdf64_add(acc, cdf64_mul(av, bv));
  }
  out_c[2u * idx] = pack_df64_f64(vec2<f32>(acc.x, acc.y));
  out_c[2u * idx + 1u] = pack_df64_f64(vec2<f32>(acc.z, acc.w));
}
''';

const String _kronRealShader =
    '''
$linalgDf64WgslLibrary

struct KronParams {
  total_elements: u32,
  ndim: u32,
  pad0: u32,
  pad1: u32,
  out_shape0: vec4<u32>,
  out_shape1: vec4<u32>,
  a_shape0: vec4<u32>,
  a_shape1: vec4<u32>,
  b_shape0: vec4<u32>,
  b_shape1: vec4<u32>,
};

fn get_vec8(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
  switch (d) {
    case 0u: { return v0.x; }
    case 1u: { return v0.y; }
    case 2u: { return v0.z; }
    case 3u: { return v0.w; }
    case 4u: { return v1.x; }
    case 5u: { return v1.y; }
    case 6u: { return v1.z; }
    default: { return v1.w; }
  }
}

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_c: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: KronParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total_elements) { return; }

  var rem = idx;
  var a_idx = 0u;
  var b_idx = 0u;
  var a_stride = 1u;
  var b_stride = 1u;
  if (params.ndim > 0u) {
    var d = i32(params.ndim) - 1;
    loop {
      if (d < 0) { break; }
      let ud = u32(d);
      let o_sz = max(1u, get_vec8(params.out_shape0, params.out_shape1, ud));
      let a_sz = max(1u, get_vec8(params.a_shape0, params.a_shape1, ud));
      let b_sz = max(1u, get_vec8(params.b_shape0, params.b_shape1, ud));
      let coord = rem % o_sz;
      rem = rem / o_sz;
      let ca = coord / b_sz;
      let cb = coord % b_sz;
      a_idx = a_idx + ca * a_stride;
      b_idx = b_idx + cb * b_stride;
      a_stride = a_stride * a_sz;
      b_stride = b_stride * b_sz;
      d = d - 1;
    }
  }
  let av = unpack_f64_df64(in_a[a_idx]);
  let bv = unpack_f64_df64(in_b[b_idx]);
  out_c[idx] = pack_df64_f64(df64_mul(av, bv));
}
''';

const String _crossRealShader =
    '''
$linalgDf64WgslLibrary

struct CrossParams {
  batch_count: u32,
  dim_a: u32,
  dim_b: u32,
  batch_ndim: u32,
  batch_shape0: vec4<u32>,
  batch_shape1: vec4<u32>,
  a_batch_shape0: vec4<u32>,
  a_batch_shape1: vec4<u32>,
  b_batch_shape0: vec4<u32>,
  b_batch_shape1: vec4<u32>,
};

fn get_vec8(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
  switch (d) {
    case 0u: { return v0.x; }
    case 1u: { return v0.y; }
    case 2u: { return v0.z; }
    case 3u: { return v0.w; }
    case 4u: { return v1.x; }
    case 5u: { return v1.y; }
    case 6u: { return v1.z; }
    default: { return v1.w; }
  }
}

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_c: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: CrossParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch_idx = gid.x;
  if (batch_idx >= params.batch_count) { return; }

  var a_batch = 0u;
  var b_batch = 0u;
  var a_stride = 1u;
  var b_stride = 1u;
  var rem = batch_idx;
  if (params.batch_ndim > 0u) {
    var d = i32(params.batch_ndim) - 1;
    loop {
      if (d < 0) { break; }
      let ud = u32(d);
      let b_sz = max(1u, get_vec8(params.batch_shape0, params.batch_shape1, ud));
      let a_sz = max(1u, get_vec8(params.a_batch_shape0, params.a_batch_shape1, ud));
      let bb_sz = max(1u, get_vec8(params.b_batch_shape0, params.b_batch_shape1, ud));
      let coord = rem % b_sz;
      rem = rem / b_sz;
      a_batch = a_batch + (coord % a_sz) * a_stride;
      b_batch = b_batch + (coord % bb_sz) * b_stride;
      a_stride = a_stride * a_sz;
      b_stride = b_stride * bb_sz;
      d = d - 1;
    }
  }

  let a_base = a_batch * params.dim_a;
  let b_base = b_batch * params.dim_b;
  let u0 = unpack_f64_df64(in_a[a_base]);
  let u1 = unpack_f64_df64(in_a[a_base + 1u]);
  var u2 = df64_zero();
  if (params.dim_a == 3u) {
    u2 = unpack_f64_df64(in_a[a_base + 2u]);
  }
  let v0 = unpack_f64_df64(in_b[b_base]);
  let v1 = unpack_f64_df64(in_b[b_base + 1u]);
  var v2 = df64_zero();
  if (params.dim_b == 3u) {
    v2 = unpack_f64_df64(in_b[b_base + 2u]);
  }

  let c2 = df64_sub(df64_mul(u0, v1), df64_mul(u1, v0));
  if (params.dim_a == 2u && params.dim_b == 2u) {
    out_c[batch_idx] = pack_df64_f64(c2);
  } else {
    let c0 = df64_sub(df64_mul(u1, v2), df64_mul(u2, v1));
    let c1 = df64_sub(df64_mul(u2, v0), df64_mul(u0, v2));
    out_c[batch_idx * 3u] = pack_df64_f64(c0);
    out_c[batch_idx * 3u + 1u] = pack_df64_f64(c1);
    out_c[batch_idx * 3u + 2u] = pack_df64_f64(c2);
  }
}
''';

const String _einsumRealShader =
    '''
$linalgDf64WgslLibrary

struct EinsumParams {
  out_size: u32,
  contract_size: u32,
  num_out_labels: u32,
  num_total_labels: u32,
  num_operands: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  label_sizes0: vec4<u32>,
  label_sizes1: vec4<u32>,
  label_sizes2: vec4<u32>,
  op0_strides0: vec4<u32>,
  op0_strides1: vec4<u32>,
  op0_strides2: vec4<u32>,
  op1_strides0: vec4<u32>,
  op1_strides1: vec4<u32>,
  op1_strides2: vec4<u32>,
  op2_strides0: vec4<u32>,
  op2_strides1: vec4<u32>,
  op2_strides2: vec4<u32>,
};

fn get_vec12(v0: vec4<u32>, v1: vec4<u32>, v2: vec4<u32>, d: u32) -> u32 {
  switch (d) {
    case 0u: { return v0.x; }
    case 1u: { return v0.y; }
    case 2u: { return v0.z; }
    case 3u: { return v0.w; }
    case 4u: { return v1.x; }
    case 5u: { return v1.y; }
    case 6u: { return v1.z; }
    case 7u: { return v1.w; }
    case 8u: { return v2.x; }
    case 9u: { return v2.y; }
    case 10u: { return v2.z; }
    default: { return v2.w; }
  }
}

@group(0) @binding(0) var<storage, read> in_op0: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_op1: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> in_op2: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_res: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params: EinsumParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_idx = gid.x;
  if (out_idx >= params.out_size) { return; }

  var base0 = 0u;
  var base1 = 0u;
  var base2 = 0u;
  var rem_out = out_idx;
  if (params.num_out_labels > 0u) {
    var d = i32(params.num_out_labels) - 1;
    loop {
      if (d < 0) { break; }
      let ud = u32(d);
      let sz = max(1u, get_vec12(params.label_sizes0, params.label_sizes1, params.label_sizes2, ud));
      let coord = rem_out % sz;
      rem_out = rem_out / sz;
      base0 = base0 + coord * get_vec12(params.op0_strides0, params.op0_strides1, params.op0_strides2, ud);
      base1 = base1 + coord * get_vec12(params.op1_strides0, params.op1_strides1, params.op1_strides2, ud);
      base2 = base2 + coord * get_vec12(params.op2_strides0, params.op2_strides1, params.op2_strides2, ud);
      d = d - 1;
    }
  }

  var acc = df64_zero();
  for (var c_idx = 0u; c_idx < params.contract_size; c_idx = c_idx + 1u) {
    var idx0 = base0;
    var idx1 = base1;
    var idx2 = base2;
    var rem_c = c_idx;
    if (params.num_total_labels > params.num_out_labels) {
      var d = i32(params.num_total_labels) - 1;
      loop {
        if (d < i32(params.num_out_labels)) { break; }
        let ud = u32(d);
        let sz = max(1u, get_vec12(params.label_sizes0, params.label_sizes1, params.label_sizes2, ud));
        let coord = rem_c % sz;
        rem_c = rem_c / sz;
        idx0 = idx0 + coord * get_vec12(params.op0_strides0, params.op0_strides1, params.op0_strides2, ud);
        idx1 = idx1 + coord * get_vec12(params.op1_strides0, params.op1_strides1, params.op1_strides2, ud);
        idx2 = idx2 + coord * get_vec12(params.op2_strides0, params.op2_strides1, params.op2_strides2, ud);
        d = d - 1;
      }
    }
    var prod = unpack_f64_df64(in_op0[idx0]);
    if (params.num_operands >= 2u) {
      prod = df64_mul(prod, unpack_f64_df64(in_op1[idx1]));
    }
    if (params.num_operands >= 3u) {
      prod = df64_mul(prod, unpack_f64_df64(in_op2[idx2]));
    }
    acc = df64_add(acc, prod);
  }
  out_res[out_idx] = pack_df64_f64(acc);
}
''';

List<int> _packMatmulUniforms({
  required int batchCount,
  required int m,
  required int k,
  required int n,
  required List<int> batchShape,
  required List<int> aBatchShape,
  required List<int> bBatchShape,
  required bool transposeB,
  required bool conjugateA,
}) {
  final uniforms = List<int>.filled(32, 0);
  uniforms[0] = batchCount;
  uniforms[1] = m;
  uniforms[2] = k;
  uniforms[3] = n;
  uniforms[4] = batchShape.length;
  uniforms[5] = transposeB ? 1 : 0;
  uniforms[6] = conjugateA ? 1 : 0;
  for (var i = 0; i < batchShape.length && i < 8; i++) {
    uniforms[8 + i] = batchShape[i];
    uniforms[16 + i] = aBatchShape[i];
    uniforms[24 + i] = bBatchShape[i];
  }
  return uniforms;
}

/// Dispatches a batched high-precision real (`Float64`) matrix multiplication.
GpuBuffer dispatchBatchedMatmulF64Gpu(
  GpuDevice device,
  GpuBuffer aF64,
  GpuBuffer bF64, {
  required int batchCount,
  required int m,
  required int k,
  required int n,
  List<int> batchShape = const [],
  List<int> aBatchShape = const [],
  List<int> bBatchShape = const [],
  bool transposeB = false,
}) {
  final total = batchCount * m * n;
  final outC = device.createBuffer(sizeInBytes: math.max(1, total) * 8);
  if (total == 0) return outC;

  final module = getOrCreateLinalgShader(
    'linalg_batched_matmul_f64',
    () => _batchedMatmulRealShader,
    workgroupSize: 64,
  );
  final uniforms = _packMatmulUniforms(
    batchCount: batchCount,
    m: m,
    k: k,
    n: n,
    batchShape: batchShape,
    aBatchShape: aBatchShape,
    bBatchShape: bBatchShape,
    transposeB: transposeB,
    conjugateA: false,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [aF64, bF64, outC],
    uniforms: uniforms,
    workgroupsX: (total + 63) ~/ 64,
  );
  return outC;
}

/// Dispatches a batched high-precision complex (`Complex128`) matrix
/// multiplication, with optional complex conjugation of the first operand.
GpuBuffer dispatchBatchedMatmulC128Gpu(
  GpuDevice device,
  GpuBuffer aC128,
  GpuBuffer bC128, {
  required int batchCount,
  required int m,
  required int k,
  required int n,
  List<int> batchShape = const [],
  List<int> aBatchShape = const [],
  List<int> bBatchShape = const [],
  bool transposeB = false,
  bool conjugateA = false,
}) {
  final total = batchCount * m * n;
  final outC = device.createBuffer(sizeInBytes: math.max(1, total) * 16);
  if (total == 0) return outC;

  final module = getOrCreateLinalgShader(
    'linalg_batched_matmul_c128',
    () => _batchedMatmulComplexShader,
    workgroupSize: 64,
  );
  final uniforms = _packMatmulUniforms(
    batchCount: batchCount,
    m: m,
    k: k,
    n: n,
    batchShape: batchShape,
    aBatchShape: aBatchShape,
    bBatchShape: bBatchShape,
    transposeB: transposeB,
    conjugateA: conjugateA,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [aC128, bC128, outC],
    uniforms: uniforms,
    workgroupsX: (total + 63) ~/ 64,
  );
  return outC;
}

/// Dispatches the Kronecker product kernel on [device].
GpuBuffer dispatchKronF64Gpu(
  GpuDevice device,
  GpuBuffer aF64,
  GpuBuffer bF64, {
  required List<int> outShape,
  required List<int> aShapePadded,
  required List<int> bShapePadded,
}) {
  final total = outShape.isEmpty ? 1 : outShape.reduce((a, b) => a * b);
  final outC = device.createBuffer(sizeInBytes: math.max(1, total) * 8);
  if (total == 0) return outC;

  final uniforms = List<int>.filled(28, 0);
  uniforms[0] = total;
  uniforms[1] = outShape.length;
  for (var i = 0; i < outShape.length && i < 8; i++) {
    uniforms[4 + i] = outShape[i];
    uniforms[12 + i] = aShapePadded[i];
    uniforms[20 + i] = bShapePadded[i];
  }

  final module = getOrCreateLinalgShader(
    'linalg_kron_f64',
    () => _kronRealShader,
    workgroupSize: 64,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [aF64, bF64, outC],
    uniforms: uniforms,
    workgroupsX: (total + 63) ~/ 64,
  );
  return outC;
}

/// Dispatches the vector cross product kernel on [device].
GpuBuffer dispatchCrossF64Gpu(
  GpuDevice device,
  GpuBuffer aF64,
  GpuBuffer bF64, {
  required int batchCount,
  required int dimA,
  required int dimB,
  required List<int> batchShape,
  required List<int> aBatchShape,
  required List<int> bBatchShape,
}) {
  final outVecDim = (dimA == 2 && dimB == 2) ? 1 : 3;
  final totalElements = batchCount * outVecDim;
  final outC = device.createBuffer(sizeInBytes: math.max(1, totalElements) * 8);
  if (batchCount == 0) return outC;

  final uniforms = List<int>.filled(28, 0);
  uniforms[0] = batchCount;
  uniforms[1] = dimA;
  uniforms[2] = dimB;
  uniforms[3] = batchShape.length;
  for (var i = 0; i < batchShape.length && i < 8; i++) {
    uniforms[4 + i] = batchShape[i];
    uniforms[12 + i] = aBatchShape[i];
    uniforms[20 + i] = bBatchShape[i];
  }

  final module = getOrCreateLinalgShader(
    'linalg_cross_f64',
    () => _crossRealShader,
    workgroupSize: 64,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [aF64, bF64, outC],
    uniforms: uniforms,
    workgroupsX: (batchCount + 63) ~/ 64,
  );
  return outC;
}

/// Dispatches the general Einstein summation contraction kernel on [device].
GpuBuffer dispatchEinsumF64Gpu(
  GpuDevice device,
  List<GpuBuffer> operandBuffers, {
  required int outSize,
  required int contractSize,
  required int numOutLabels,
  required int numTotalLabels,
  required List<int> labelSizes,
  required List<List<int>> operandLabelStrides,
}) {
  final outRes = device.createBuffer(sizeInBytes: math.max(1, outSize) * 8);
  if (outSize == 0) return outRes;

  final op0 = operandBuffers[0];
  final op1 = operandBuffers.length >= 2 ? operandBuffers[1] : op0;
  final op2 = operandBuffers.length >= 3 ? operandBuffers[2] : op0;

  final uniforms = List<int>.filled(56, 0);
  uniforms[0] = outSize;
  uniforms[1] = math.max(1, contractSize);
  uniforms[2] = numOutLabels;
  uniforms[3] = numTotalLabels;
  uniforms[4] = operandBuffers.length;
  for (var i = 0; i < numTotalLabels && i < 12; i++) {
    uniforms[8 + i] = labelSizes[i];
    uniforms[20 + i] = operandLabelStrides[0][i];
    if (operandLabelStrides.length >= 2) {
      uniforms[32 + i] = operandLabelStrides[1][i];
    }
    if (operandLabelStrides.length >= 3) {
      uniforms[44 + i] = operandLabelStrides[2][i];
    }
  }

  final module = getOrCreateLinalgShader(
    'linalg_einsum_f64',
    () => _einsumRealShader,
    workgroupSize: 64,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [op0, op1, op2, outRes],
    uniforms: uniforms,
    workgroupsX: (outSize + 63) ~/ 64,
  );
  return outRes;
}
