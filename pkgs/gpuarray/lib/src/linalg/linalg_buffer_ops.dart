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
import '../dtype.dart';
import '../gpu_array.dart';
import 'linalg_wgsl_df64.dart';

/// Validates an optional destination [out] array against [expectedDevice],
/// [expectedShape], and [expectedDType].
void validateLinalgOut<T extends DTypeTag>(
  GpuArray<T>? out,
  GpuDevice expectedDevice,
  List<int> expectedShape,
  DType expectedDType, {
  String paramName = 'out',
}) {
  if (out == null) return;
  if (out.isDisposed) {
    throw StateError('Cannot use a disposed GpuArray as output ($paramName).');
  }
  if (out.device != expectedDevice) {
    throw ArgumentError.value(
      out,
      paramName,
      'Must reside on the same GpuDevice as the input array.',
    );
  }
  if (out.dtype != expectedDType) {
    throw ArgumentError.value(
      out,
      paramName,
      'Must have dtype $expectedDType, got ${out.dtype}.',
    );
  }
  if (out.size > 1 && out.strides.contains(0)) {
    throw ArgumentError.value(
      out,
      paramName,
      'Must be writeable and not a broadcasted view.',
    );
  }
  if (out.shape.length != expectedShape.length) {
    throw ArgumentError.value(
      out,
      paramName,
      'Must have shape $expectedShape, got ${out.shape}.',
    );
  }
  for (var i = 0; i < expectedShape.length; i++) {
    if (out.shape[i] != expectedShape[i]) {
      throw ArgumentError.value(
        out,
        paramName,
        'Must have shape $expectedShape, got ${out.shape}.',
      );
    }
  }
}

/// Maps [dtype] to an integer code for WGSL shader dispatch.
int dtypeCodeForWgsl(DType dtype) {
  switch (dtype) {
    case DType.float64:
      return 0;
    case DType.float32:
      return 1;
    case DType.float16:
      return 2;
    case DType.bfloat16:
      return 3;
    case DType.int64:
      return 4;
    case DType.int32:
      return 5;
    case DType.int16:
      return 6;
    case DType.int8:
      return 7;
    case DType.uint64:
      return 8;
    case DType.uint32:
      return 9;
    case DType.uint16:
      return 10;
    case DType.uint8:
      return 11;
    case DType.boolean:
      return 12;
    case DType.complex64:
      return 13;
    case DType.complex128:
      return 14;
  }
}

List<int> _packStridedUniforms(
  int elementCount,
  int ndim,
  int offsetElements,
  DType dtype,
  List<int> shape,
  List<int> elementStrides,
) {
  final byteWidth = dtype.byteWidth;
  final uniforms = List<int>.filled(36, 0);
  uniforms[0] = elementCount;
  uniforms[1] = ndim;
  uniforms[2] = offsetElements * byteWidth;
  uniforms[3] = dtypeCodeForWgsl(dtype);
  for (var i = 0; i < ndim && i < 16; i++) {
    uniforms[4 + i] = shape[i];
    uniforms[20 + i] = elementStrides[i] * byteWidth;
  }
  return uniforms;
}

const String _stridedIndexWgsl = r'''
struct StridedParams {
  total_elements: u32,
  ndim: u32,
  offset_bytes: i32,
  dtype_code: u32,
  shape0: vec4<u32>,
  shape1: vec4<u32>,
  shape2: vec4<u32>,
  shape3: vec4<u32>,
  stride0: vec4<i32>,
  stride1: vec4<i32>,
  stride2: vec4<i32>,
  stride3: vec4<i32>,
};

fn get_dim_size(p: StridedParams, d: u32) -> u32 {
  switch (d) {
    case 0u: { return p.shape0.x; }
    case 1u: { return p.shape0.y; }
    case 2u: { return p.shape0.z; }
    case 3u: { return p.shape0.w; }
    case 4u: { return p.shape1.x; }
    case 5u: { return p.shape1.y; }
    case 6u: { return p.shape1.z; }
    case 7u: { return p.shape1.w; }
    case 8u: { return p.shape2.x; }
    case 9u: { return p.shape2.y; }
    case 10u: { return p.shape2.z; }
    case 11u: { return p.shape2.w; }
    case 12u: { return p.shape3.x; }
    case 13u: { return p.shape3.y; }
    case 14u: { return p.shape3.z; }
    default: { return p.shape3.w; }
  }
}

fn get_dim_stride(p: StridedParams, d: u32) -> i32 {
  switch (d) {
    case 0u: { return p.stride0.x; }
    case 1u: { return p.stride0.y; }
    case 2u: { return p.stride0.z; }
    case 3u: { return p.stride0.w; }
    case 4u: { return p.stride1.x; }
    case 5u: { return p.stride1.y; }
    case 6u: { return p.stride1.z; }
    case 7u: { return p.stride1.w; }
    case 8u: { return p.stride2.x; }
    case 9u: { return p.stride2.y; }
    case 10u: { return p.stride2.z; }
    case 11u: { return p.stride2.w; }
    case 12u: { return p.stride3.x; }
    case 13u: { return p.stride3.y; }
    case 14u: { return p.stride3.z; }
    default: { return p.stride3.w; }
  }
}

fn compute_byte_offset(p: StridedParams, flat_index: u32) -> u32 {
  var rem = flat_index;
  var byte_off = p.offset_bytes;
  if (p.ndim > 0u) {
    var d = i32(p.ndim) - 1;
    loop {
      if (d < 0) { break; }
      let dim_sz = max(1u, get_dim_size(p, u32(d)));
      let coord = rem % dim_sz;
      rem = rem / dim_sz;
      byte_off = byte_off + i32(coord) * get_dim_stride(p, u32(d));
      d = d - 1;
    }
  }
  return u32(max(0, byte_off));
}

fn u64_to_df64(lo: u32, hi: u32) -> vec2<f32> {
  let lo_hi = df64_from_f32(f32(lo >> 16u) * 65536.0);
  let lo_lo = df64_from_f32(f32(lo & 0xFFFFu));
  let hi_hi = df64_from_f32(f32(hi >> 16u) * 281474976710656.0);
  let hi_lo = df64_from_f32(f32(hi & 0xFFFFu) * 4294967296.0);
  return df64_add(df64_add(hi_hi, hi_lo), df64_add(lo_hi, lo_lo));
}

fn i64_to_df64(lo: u32, hi: u32) -> vec2<f32> {
  if ((hi & 0x80000000u) != 0u) {
    let n_lo = (~lo) + 1u;
    let carry = select(0u, 1u, n_lo == 0u);
    let n_hi = (~hi) + carry;
    return df64_neg(u64_to_df64(n_lo, n_hi));
  }
  return u64_to_df64(lo, hi);
}
''';

const String _toContiguousF64Shader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_f64: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: StridedParams;

fn read_as_f64_bits(u_byte: u32, code: u32) -> vec2<u32> {
  let word_idx = u_byte >> 2u;
  let byte_shift = (u_byte & 3u) * 8u;
  switch (code) {
    case 0u: {
      return vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]);
    }
    case 1u: {
      return pack_df64_f64(df64_from_f32(bitcast<f32>(input_words[word_idx])));
    }
    case 2u: {
      let half_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      return pack_df64_f64(df64_from_f32(unpack2x16float(half_bits).x));
    }
    case 3u: {
      let bf_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      return pack_df64_f64(df64_from_f32(bitcast<f32>(bf_bits << 16u)));
    }
    case 4u: {
      return pack_df64_f64(i64_to_df64(input_words[word_idx], input_words[word_idx + 1u]));
    }
    case 5u: {
      let s = bitcast<i32>(input_words[word_idx]);
      let hi = s / 65536;
      let lo = s - hi * 65536;
      return pack_df64_f64(df64_add(df64_from_f32(f32(hi) * 65536.0), df64_from_f32(f32(lo))));
    }
    case 6u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      let s = (bitcast<i32>(raw) << 16) >> 16;
      return pack_df64_f64(df64_from_f32(f32(s)));
    }
    case 7u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      let s = (bitcast<i32>(raw) << 24) >> 24;
      return pack_df64_f64(df64_from_f32(f32(s)));
    }
    case 8u: {
      return pack_df64_f64(u64_to_df64(input_words[word_idx], input_words[word_idx + 1u]));
    }
    case 9u: {
      let u = input_words[word_idx];
      let hi = u >> 16u;
      let lo = u & 0xFFFFu;
      return pack_df64_f64(df64_add(df64_from_f32(f32(hi) * 65536.0), df64_from_f32(f32(lo))));
    }
    case 10u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      return pack_df64_f64(df64_from_f32(f32(raw)));
    }
    case 11u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      return pack_df64_f64(df64_from_f32(f32(raw)));
    }
    case 12u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      return pack_df64_f64(df64_from_f32(select(0.0, 1.0, raw != 0u)));
    }
    case 13u: {
      return pack_df64_f64(df64_from_f32(bitcast<f32>(input_words[word_idx])));
    }
    default: {
      return vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]);
    }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let u_byte = compute_byte_offset(params, i);
  out_f64[i] = read_as_f64_bits(u_byte, params.dtype_code);
}
''';

const String _toContiguousC128Shader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_c128: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: StridedParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  let byte_shift = (u_byte & 3u) * 8u;
  var re_bits = vec2<u32>(0u, 0u);
  var im_bits = vec2<u32>(0u, 0u);
  switch (params.dtype_code) {
    case 14u: {
      re_bits = vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]);
      im_bits = vec2<u32>(input_words[word_idx + 2u], input_words[word_idx + 3u]);
    }
    case 13u: {
      re_bits = pack_df64_f64(df64_from_f32(bitcast<f32>(input_words[word_idx])));
      im_bits = pack_df64_f64(df64_from_f32(bitcast<f32>(input_words[word_idx + 1u])));
    }
    case 0u: {
      re_bits = vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]);
    }
    case 1u: {
      re_bits = pack_df64_f64(df64_from_f32(bitcast<f32>(input_words[word_idx])));
    }
    case 2u: {
      let half_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      re_bits = pack_df64_f64(df64_from_f32(unpack2x16float(half_bits).x));
    }
    case 3u: {
      let bf_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      re_bits = pack_df64_f64(df64_from_f32(bitcast<f32>(bf_bits << 16u)));
    }
    case 4u: {
      re_bits = pack_df64_f64(i64_to_df64(input_words[word_idx], input_words[word_idx + 1u]));
    }
    case 5u: {
      let s = bitcast<i32>(input_words[word_idx]);
      let hi = s / 65536;
      let lo = s - hi * 65536;
      re_bits = pack_df64_f64(df64_add(df64_from_f32(f32(hi) * 65536.0), df64_from_f32(f32(lo))));
    }
    case 6u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      let s = (bitcast<i32>(raw) << 16) >> 16;
      re_bits = pack_df64_f64(df64_from_f32(f32(s)));
    }
    case 7u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      let s = (bitcast<i32>(raw) << 24) >> 24;
      re_bits = pack_df64_f64(df64_from_f32(f32(s)));
    }
    case 8u: {
      re_bits = pack_df64_f64(u64_to_df64(input_words[word_idx], input_words[word_idx + 1u]));
    }
    case 9u: {
      let u = input_words[word_idx];
      let hi = u >> 16u;
      let lo = u & 0xFFFFu;
      re_bits = pack_df64_f64(df64_add(df64_from_f32(f32(hi) * 65536.0), df64_from_f32(f32(lo))));
    }
    case 10u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      re_bits = pack_df64_f64(df64_from_f32(f32(raw)));
    }
    case 11u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      re_bits = pack_df64_f64(df64_from_f32(f32(raw)));
    }
    default: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      re_bits = pack_df64_f64(df64_from_f32(select(0.0, 1.0, raw != 0u)));
    }
  }
  out_c128[2u * i] = re_bits;
  out_c128[2u * i + 1u] = im_bits;
}
''';

const String _fromF64ToArrayWordShader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> in_f64: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(2) var<uniform> params: StridedParams;

fn df64_to_i32_exact(v: vec2<f32>) -> i32 {
  let hi = i32(v.x);
  let rem = (v.x - f32(hi)) + v.y;
  return hi + i32(round(rem));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let bits = in_f64[i];
  let v = unpack_f64_df64(bits);
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  switch (params.dtype_code) {
    case 0u: {
      out_words[word_idx] = bits.x;
      out_words[word_idx + 1u] = bits.y;
    }
    case 1u: {
      out_words[word_idx] = bitcast<u32>(v.x + v.y);
    }
    case 4u: {
      let iv = df64_to_i32_exact(v);
      out_words[word_idx] = bitcast<u32>(iv);
      out_words[word_idx + 1u] = select(0u, 0xFFFFFFFFu, iv < 0);
    }
    case 5u: {
      out_words[word_idx] = bitcast<u32>(df64_to_i32_exact(v));
    }
    case 8u: {
      let uv = u32(max(0.0, round(v.x + v.y)));
      out_words[word_idx] = uv;
      out_words[word_idx + 1u] = 0u;
    }
    case 9u: {
      out_words[word_idx] = u32(max(0.0, round(v.x + v.y)));
    }
    case 13u: {
      out_words[word_idx] = bitcast<u32>(v.x + v.y);
      out_words[word_idx + 1u] = 0u;
    }
    default: {
      out_words[word_idx] = bits.x;
      out_words[word_idx + 1u] = bits.y;
      out_words[word_idx + 2u] = 0u;
      out_words[word_idx + 3u] = 0u;
    }
  }
}
''';

const String _fromF64ToArraySubwordShader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> in_f64: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out_words: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: StridedParams;

fn atomic_write_subword(word_idx: u32, shift: u32, mask: u32, value: u32) {
  let shifted_mask = mask << shift;
  let shifted_val = (value & mask) << shift;
  var old_word = atomicLoad(&out_words[word_idx]);
  loop {
    let new_word = (old_word & (~shifted_mask)) | shifted_val;
    let res = atomicCompareExchangeWeak(&out_words[word_idx], old_word, new_word);
    if (res.exchanged) { break; }
    old_word = res.old_value;
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let v = unpack_f64_df64(in_f64[i]);
  let f = v.x + v.y;
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  let shift = (u_byte & 3u) * 8u;
  switch (params.dtype_code) {
    case 2u: {
      let h = pack2x16float(vec2<f32>(f, 0.0)) & 0xFFFFu;
      atomic_write_subword(word_idx, shift, 0xFFFFu, h);
    }
    case 3u: {
      let bf = (bitcast<u32>(f) + 0x8000u) >> 16u;
      atomic_write_subword(word_idx, shift, 0xFFFFu, bf);
    }
    case 6u: {
      let iv = bitcast<u32>(i32(round(f))) & 0xFFFFu;
      atomic_write_subword(word_idx, shift, 0xFFFFu, iv);
    }
    case 7u: {
      let iv = bitcast<u32>(i32(round(f))) & 0xFFu;
      atomic_write_subword(word_idx, shift, 0xFFu, iv);
    }
    case 10u: {
      let uv = u32(max(0.0, round(f))) & 0xFFFFu;
      atomic_write_subword(word_idx, shift, 0xFFFFu, uv);
    }
    case 11u: {
      let uv = u32(max(0.0, round(f))) & 0xFFu;
      atomic_write_subword(word_idx, shift, 0xFFu, uv);
    }
    default: {
      let bv = select(0u, 1u, f != 0.0);
      atomic_write_subword(word_idx, shift, 0xFFu, bv);
    }
  }
}
''';

const String _fromC128ToArrayShader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> in_c128: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(2) var<uniform> params: StridedParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let re_bits = in_c128[2u * i];
  let im_bits = in_c128[2u * i + 1u];
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  switch (params.dtype_code) {
    case 14u: {
      out_words[word_idx] = re_bits.x;
      out_words[word_idx + 1u] = re_bits.y;
      out_words[word_idx + 2u] = im_bits.x;
      out_words[word_idx + 3u] = im_bits.y;
    }
    case 13u: {
      let re = unpack_f64_df64(re_bits);
      let im = unpack_f64_df64(im_bits);
      out_words[word_idx] = bitcast<u32>(re.x + re.y);
      out_words[word_idx + 1u] = bitcast<u32>(im.x + im.y);
    }
    case 0u: {
      out_words[word_idx] = re_bits.x;
      out_words[word_idx + 1u] = re_bits.y;
    }
    default: {
      let re = unpack_f64_df64(re_bits);
      out_words[word_idx] = bitcast<u32>(re.x + re.y);
    }
  }
}
''';

const String _toContiguousF32Shader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_f32: array<f32>;
@group(0) @binding(2) var<uniform> params: StridedParams;

fn read_as_f32(u_byte: u32, code: u32) -> f32 {
  let word_idx = u_byte >> 2u;
  let byte_shift = (u_byte & 3u) * 8u;
  switch (code) {
    case 0u: {
      let v = unpack_f64_df64(vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]));
      return v.x + v.y;
    }
    case 1u: {
      return bitcast<f32>(input_words[word_idx]);
    }
    case 2u: {
      let half_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      return unpack2x16float(half_bits).x;
    }
    case 3u: {
      let bf_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      return bitcast<f32>(bf_bits << 16u);
    }
    case 4u: {
      let v = i64_to_df64(input_words[word_idx], input_words[word_idx + 1u]);
      return v.x + v.y;
    }
    case 5u: {
      return f32(bitcast<i32>(input_words[word_idx]));
    }
    case 6u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      return f32((bitcast<i32>(raw) << 16) >> 16);
    }
    case 7u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      return f32((bitcast<i32>(raw) << 24) >> 24);
    }
    case 8u: {
      let v = u64_to_df64(input_words[word_idx], input_words[word_idx + 1u]);
      return v.x + v.y;
    }
    case 9u: {
      return f32(input_words[word_idx]);
    }
    case 10u: {
      return f32((input_words[word_idx] >> byte_shift) & 0xFFFFu);
    }
    case 11u: {
      return f32((input_words[word_idx] >> byte_shift) & 0xFFu);
    }
    case 12u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      return select(0.0, 1.0, raw != 0u);
    }
    case 13u: {
      return bitcast<f32>(input_words[word_idx]);
    }
    default: {
      let v = unpack_f64_df64(vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]));
      return v.x + v.y;
    }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let u_byte = compute_byte_offset(params, i);
  out_f32[i] = read_as_f32(u_byte, params.dtype_code);
}
''';

const String _toContiguousC64Shader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_c64: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: StridedParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  let byte_shift = (u_byte & 3u) * 8u;
  var re = 0.0;
  var im = 0.0;
  switch (params.dtype_code) {
    case 14u: {
      let re_v = unpack_f64_df64(vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]));
      let im_v = unpack_f64_df64(vec2<u32>(input_words[word_idx + 2u], input_words[word_idx + 3u]));
      re = re_v.x + re_v.y;
      im = im_v.x + im_v.y;
    }
    case 13u: {
      re = bitcast<f32>(input_words[word_idx]);
      im = bitcast<f32>(input_words[word_idx + 1u]);
    }
    case 0u: {
      let re_v = unpack_f64_df64(vec2<u32>(input_words[word_idx], input_words[word_idx + 1u]));
      re = re_v.x + re_v.y;
    }
    case 1u: {
      re = bitcast<f32>(input_words[word_idx]);
    }
    case 2u: {
      let half_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      re = unpack2x16float(half_bits).x;
    }
    case 3u: {
      let bf_bits = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      re = bitcast<f32>(bf_bits << 16u);
    }
    case 4u: {
      let v = i64_to_df64(input_words[word_idx], input_words[word_idx + 1u]);
      re = v.x + v.y;
    }
    case 5u: {
      re = f32(bitcast<i32>(input_words[word_idx]));
    }
    case 6u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFFFu;
      re = f32((bitcast<i32>(raw) << 16) >> 16);
    }
    case 7u: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      re = f32((bitcast<i32>(raw) << 24) >> 24);
    }
    case 8u: {
      let v = u64_to_df64(input_words[word_idx], input_words[word_idx + 1u]);
      re = v.x + v.y;
    }
    case 9u: {
      re = f32(input_words[word_idx]);
    }
    case 10u: {
      re = f32((input_words[word_idx] >> byte_shift) & 0xFFFFu);
    }
    case 11u: {
      re = f32((input_words[word_idx] >> byte_shift) & 0xFFu);
    }
    default: {
      let raw = (input_words[word_idx] >> byte_shift) & 0xFFu;
      re = select(0.0, 1.0, raw != 0u);
    }
  }
  out_c64[i] = vec2<f32>(re, im);
}
''';

const String _fromF32ToArrayWordShader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> in_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(2) var<uniform> params: StridedParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let f = in_f32[i];
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  switch (params.dtype_code) {
    case 1u: {
      out_words[word_idx] = bitcast<u32>(f);
    }
    case 0u: {
      let bits = pack_df64_f64(df64_from_f32(f));
      out_words[word_idx] = bits.x;
      out_words[word_idx + 1u] = bits.y;
    }
    case 4u: {
      let iv = i32(round(f));
      out_words[word_idx] = bitcast<u32>(iv);
      out_words[word_idx + 1u] = select(0u, 0xFFFFFFFFu, iv < 0);
    }
    case 5u: {
      out_words[word_idx] = bitcast<u32>(i32(round(f)));
    }
    case 8u: {
      let uv = u32(max(0.0, round(f)));
      out_words[word_idx] = uv;
      out_words[word_idx + 1u] = 0u;
    }
    case 9u: {
      out_words[word_idx] = u32(max(0.0, round(f)));
    }
    case 13u: {
      out_words[word_idx] = bitcast<u32>(f);
      out_words[word_idx + 1u] = 0u;
    }
    default: {
      let bits = pack_df64_f64(df64_from_f32(f));
      out_words[word_idx] = bits.x;
      out_words[word_idx + 1u] = bits.y;
      out_words[word_idx + 2u] = 0u;
      out_words[word_idx + 3u] = 0u;
    }
  }
}
''';

const String _fromF32ToArraySubwordShader =
    '''
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> in_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_words: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: StridedParams;

fn atomic_write_subword(word_idx: u32, shift: u32, mask: u32, value: u32) {
  let shifted_mask = mask << shift;
  let shifted_val = (value & mask) << shift;
  var old_word = atomicLoad(&out_words[word_idx]);
  loop {
    let new_word = (old_word & (~shifted_mask)) | shifted_val;
    let res = atomicCompareExchangeWeak(&out_words[word_idx], old_word, new_word);
    if (res.exchanged) { break; }
    old_word = res.old_value;
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let f = in_f32[i];
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  let shift = (u_byte & 3u) * 8u;
  switch (params.dtype_code) {
    case 2u: {
      let h = pack2x16float(vec2<f32>(f, 0.0)) & 0xFFFFu;
      atomic_write_subword(word_idx, shift, 0xFFFFu, h);
    }
    case 3u: {
      let bf = (bitcast<u32>(f) + 0x8000u) >> 16u;
      atomic_write_subword(word_idx, shift, 0xFFFFu, bf);
    }
    case 6u: {
      let iv = bitcast<u32>(i32(round(f))) & 0xFFFFu;
      atomic_write_subword(word_idx, shift, 0xFFFFu, iv);
    }
    case 7u: {
      let iv = bitcast<u32>(i32(round(f))) & 0xFFu;
      atomic_write_subword(word_idx, shift, 0xFFu, iv);
    }
    case 10u: {
      let uv = u32(max(0.0, round(f))) & 0xFFFFu;
      atomic_write_subword(word_idx, shift, 0xFFFFu, uv);
    }
    case 11u: {
      let uv = u32(max(0.0, round(f))) & 0xFFu;
      atomic_write_subword(word_idx, shift, 0xFFu, uv);
    }
    default: {
      let bv = select(0u, 1u, f != 0.0);
      atomic_write_subword(word_idx, shift, 0xFFu, bv);
    }
  }
}
''';

const String _fromC64ToArrayShader =
    '''
$linalgDf64WgslLibrary
$_stridedIndexWgsl

@group(0) @binding(0) var<storage, read> in_c64: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(2) var<uniform> params: StridedParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total_elements) { return; }
  let z = in_c64[i];
  let u_byte = compute_byte_offset(params, i);
  let word_idx = u_byte >> 2u;
  switch (params.dtype_code) {
    case 13u: {
      out_words[word_idx] = bitcast<u32>(z.x);
      out_words[word_idx + 1u] = bitcast<u32>(z.y);
    }
    case 14u: {
      let re_bits = pack_df64_f64(df64_from_f32(z.x));
      let im_bits = pack_df64_f64(df64_from_f32(z.y));
      out_words[word_idx] = re_bits.x;
      out_words[word_idx + 1u] = re_bits.y;
      out_words[word_idx + 2u] = im_bits.x;
      out_words[word_idx + 3u] = im_bits.y;
    }
    case 0u: {
      let re_bits = pack_df64_f64(df64_from_f32(z.x));
      out_words[word_idx] = re_bits.x;
      out_words[word_idx + 1u] = re_bits.y;
    }
    default: {
      out_words[word_idx] = bitcast<u32>(z.x);
    }
  }
}
''';

/// Allocates and populates a contiguous `Float32` (`array<f32>`) [GpuBuffer]
/// from [input] on GPU.
GpuBuffer toContiguousFloat32Buffer(GpuArray<DTypeTag> input) {
  final count = input.size;
  final byteLength = math.max(1, count) * 4;
  final resultBuffer = input.device.createBuffer(sizeInBytes: byteLength);
  if (count == 0) return resultBuffer;

  final module = getOrCreateLinalgShader(
    'linalg_to_f32_contiguous',
    () => _toContiguousF32Shader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    input.ndim,
    input.offsetElements,
    input.dtype,
    input.shape,
    input.strides,
  );
  input.device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [input.buffer, resultBuffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return resultBuffer;
}

/// Allocates and populates a contiguous `Complex64` (`array<vec2<f32>>`)
/// [GpuBuffer] from [input] on GPU.
GpuBuffer toContiguousComplex64Buffer(GpuArray<DTypeTag> input) {
  final count = input.size;
  final byteLength = math.max(1, count) * 8;
  final resultBuffer = input.device.createBuffer(sizeInBytes: byteLength);
  if (count == 0) return resultBuffer;

  final module = getOrCreateLinalgShader(
    'linalg_to_c64_contiguous',
    () => _toContiguousC64Shader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    input.ndim,
    input.offsetElements,
    input.dtype,
    input.shape,
    input.strides,
  );
  input.device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [input.buffer, resultBuffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return resultBuffer;
}

/// Allocates and populates a contiguous `Float64` (`array<vec2<u32>>`)
/// [GpuBuffer] from [input] on GPU.
GpuBuffer toContiguousFloat64Buffer(GpuArray<DTypeTag> input) {
  final count = input.size;
  final byteLength = math.max(1, count) * 8;
  final resultBuffer = input.device.createBuffer(sizeInBytes: byteLength);
  if (count == 0) return resultBuffer;

  final module = getOrCreateLinalgShader(
    'linalg_to_f64_contiguous',
    () => _toContiguousF64Shader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    input.ndim,
    input.offsetElements,
    input.dtype,
    input.shape,
    input.strides,
  );
  input.device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [input.buffer, resultBuffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return resultBuffer;
}

/// Allocates and populates a contiguous `Complex128` (`array<vec2<u32>>` of
/// length `2 * size`) [GpuBuffer] from [input] on GPU.
GpuBuffer toContiguousComplex128Buffer(GpuArray<DTypeTag> input) {
  final count = input.size;
  final byteLength = math.max(1, count) * 16;
  final resultBuffer = input.device.createBuffer(sizeInBytes: byteLength);
  if (count == 0) return resultBuffer;

  final module = getOrCreateLinalgShader(
    'linalg_to_c128_contiguous',
    () => _toContiguousC128Shader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    input.ndim,
    input.offsetElements,
    input.dtype,
    input.shape,
    input.strides,
  );
  input.device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [input.buffer, resultBuffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return resultBuffer;
}

/// Writes a contiguous `Float32` [sourceF32] buffer into [out] (if provided) or
/// a newly allocated [GpuArray] of [shape] and [targetDType] on [device].
GpuArray<T> writeFloat32BufferToArray<T extends DTypeTag>(
  GpuDevice device,
  GpuBuffer sourceF32,
  List<int> shape,
  DType targetDType, {
  GpuArray<T>? out,
  String outParamName = 'out',
}) {
  validateLinalgOut(out, device, shape, targetDType, paramName: outParamName);
  final destination =
      out ?? GpuArray<T>.empty(shape, targetDType as DType<T>, device: device);
  final count = destination.size;
  if (count == 0) return destination;

  final isSubword = switch (targetDType) {
    DType.float16 ||
    DType.bfloat16 ||
    DType.int16 ||
    DType.int8 ||
    DType.uint16 ||
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

  final module = getOrCreateLinalgShader(
    isSubword ? 'linalg_from_f32_subword' : 'linalg_from_f32_word',
    () => isSubword ? _fromF32ToArraySubwordShader : _fromF32ToArrayWordShader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    destination.ndim,
    destination.offsetElements,
    targetDType,
    destination.shape,
    destination.strides,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [sourceF32, destination.buffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return destination;
}

/// Writes a contiguous `Complex64` [sourceC64] buffer into [out] (if provided)
/// or a newly allocated [GpuArray] of [shape] and [targetDType] on [device].
GpuArray<T> writeComplex64BufferToArray<T extends DTypeTag>(
  GpuDevice device,
  GpuBuffer sourceC64,
  List<int> shape,
  DType targetDType, {
  GpuArray<T>? out,
  String outParamName = 'out',
}) {
  validateLinalgOut(out, device, shape, targetDType, paramName: outParamName);
  final destination =
      out ?? GpuArray<T>.empty(shape, targetDType as DType<T>, device: device);
  final count = destination.size;
  if (count == 0) return destination;

  final module = getOrCreateLinalgShader(
    'linalg_from_c64_word',
    () => _fromC64ToArrayShader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    destination.ndim,
    destination.offsetElements,
    targetDType,
    destination.shape,
    destination.strides,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [sourceC64, destination.buffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return destination;
}

/// Writes a contiguous `Float64` [sourceF64] buffer into [out] (if provided) or
/// a newly allocated [GpuArray] of [shape] and [targetDType] on [device].
GpuArray<T> writeFloat64BufferToArray<T extends DTypeTag>(
  GpuDevice device,
  GpuBuffer sourceF64,
  List<int> shape,
  DType targetDType, {
  GpuArray<T>? out,
  String outParamName = 'out',
}) {
  validateLinalgOut(out, device, shape, targetDType, paramName: outParamName);
  final destination =
      out ?? GpuArray<T>.empty(shape, targetDType as DType<T>, device: device);
  final count = destination.size;
  if (count == 0) return destination;

  final isSubword = switch (targetDType) {
    DType.float16 ||
    DType.bfloat16 ||
    DType.int16 ||
    DType.int8 ||
    DType.uint16 ||
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

  final module = getOrCreateLinalgShader(
    isSubword ? 'linalg_from_f64_subword' : 'linalg_from_f64_word',
    () => isSubword ? _fromF64ToArraySubwordShader : _fromF64ToArrayWordShader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    destination.ndim,
    destination.offsetElements,
    targetDType,
    destination.shape,
    destination.strides,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [sourceF64, destination.buffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return destination;
}

/// Writes a contiguous `Complex128` [sourceC128] buffer into [out] (if
/// provided) or a newly allocated [GpuArray] of [shape] and [targetDType].
GpuArray<T> writeComplex128BufferToArray<T extends DTypeTag>(
  GpuDevice device,
  GpuBuffer sourceC128,
  List<int> shape,
  DType targetDType, {
  GpuArray<T>? out,
  String outParamName = 'out',
}) {
  validateLinalgOut(out, device, shape, targetDType, paramName: outParamName);
  final destination =
      out ?? GpuArray<T>.empty(shape, targetDType as DType<T>, device: device);
  final count = destination.size;
  if (count == 0) return destination;

  final module = getOrCreateLinalgShader(
    'linalg_from_c128_word',
    () => _fromC128ToArrayShader,
    workgroupSize: 64,
  );
  final uniforms = _packStridedUniforms(
    count,
    destination.ndim,
    destination.offsetElements,
    targetDType,
    destination.shape,
    destination.strides,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [sourceC128, destination.buffer],
    uniforms: uniforms,
    workgroupsX: (count + 63) ~/ 64,
  );
  return destination;
}

/// Copies [source] into [out] (if provided) or a newly allocated contiguous
/// [GpuArray] on GPU.
GpuArray<T> copyGpuArray<T extends DTypeTag>(
  GpuArray<T> source, {
  GpuArray<T>? out,
  String outParamName = 'out',
}) {
  validateLinalgOut(
    out,
    source.device,
    source.shape,
    source.dtype,
    paramName: outParamName,
  );
  return ResourceScope.scope(() {
    if (isComplexDType(source.dtype)) {
      if (isSinglePrecisionDType(source.dtype)) {
        final buffer = toContiguousComplex64Buffer(source);
        final output = writeComplex64BufferToArray<T>(
          source.device,
          buffer,
          source.shape,
          source.dtype,
          out: out,
          outParamName: outParamName,
        );
        if (out == null) output.detachToParentScope();
        return output;
      }
      final buffer = toContiguousComplex128Buffer(source);
      final output = writeComplex128BufferToArray<T>(
        source.device,
        buffer,
        source.shape,
        source.dtype,
        out: out,
        outParamName: outParamName,
      );
      if (out == null) output.detachToParentScope();
      return output;
    } else {
      if (isSinglePrecisionDType(source.dtype)) {
        final buffer = toContiguousFloat32Buffer(source);
        final output = writeFloat32BufferToArray<T>(
          source.device,
          buffer,
          source.shape,
          source.dtype,
          out: out,
          outParamName: outParamName,
        );
        if (out == null) output.detachToParentScope();
        return output;
      }
      final buffer = toContiguousFloat64Buffer(source);
      final output = writeFloat64BufferToArray<T>(
        source.device,
        buffer,
        source.shape,
        source.dtype,
        out: out,
        outParamName: outParamName,
      );
      if (out == null) output.detachToParentScope();
      return output;
    }
  });
}

const String _sumLastAxisF32Shader = r'''
struct SumParams {
  outer_size: u32,
  axis_len: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> in_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_f32: array<f32>;
@group(0) @binding(2) var<uniform> params: SumParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.outer_size) { return; }
  var acc: f32 = 0.0;
  let base = row * params.axis_len;
  for (var j: u32 = 0u; j < params.axis_len; j = j + 1u) {
    acc = acc + in_f32[base + j];
  }
  out_f32[row] = acc;
}
''';

const String _sumLastAxisC64Shader = r'''
struct SumParams {
  outer_size: u32,
  axis_len: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> in_c64: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> out_c64: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: SumParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.outer_size) { return; }
  var acc = vec2<f32>(0.0, 0.0);
  let base = row * params.axis_len;
  for (var j: u32 = 0u; j < params.axis_len; j = j + 1u) {
    acc = acc + in_c64[base + j];
  }
  out_c64[row] = acc;
}
''';

const String _sumLastAxisF64Shader =
    '''
$linalgDf64WgslLibrary

struct SumParams {
  outer_size: u32,
  axis_len: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> in_f64: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out_f64: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: SumParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.outer_size) { return; }
  var acc = df64_zero();
  let base = row * params.axis_len;
  for (var j: u32 = 0u; j < params.axis_len; j = j + 1u) {
    acc = df64_add(acc, unpack_f64_df64(in_f64[base + j]));
  }
  out_f64[row] = pack_df64_f64(acc);
}
''';

const String _sumLastAxisC128Shader =
    '''
$linalgDf64WgslLibrary

struct SumParams {
  outer_size: u32,
  axis_len: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> in_c128: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out_c128: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: SumParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.outer_size) { return; }
  var acc = cdf64_zero();
  let base = row * params.axis_len;
  for (var j: u32 = 0u; j < params.axis_len; j = j + 1u) {
    let idx = (base + j) * 2u;
    let re = unpack_f64_df64(in_c128[idx]);
    let im = unpack_f64_df64(in_c128[idx + 1u]);
    let val = vec4<f32>(re.x, re.y, im.x, im.y);
    acc = cdf64_add(acc, val);
  }
  let out_idx = row * 2u;
  out_c128[out_idx] = pack_df64_f64(acc.xy);
  out_c128[out_idx + 1u] = pack_df64_f64(acc.zw);
}
''';

/// Sums [source] along its last axis on GPU using native `f32`/`c64` or
/// `df64`/`cdf64` precision.
GpuArray<T> sumLastAxisGpu<T extends DTypeTag>(
  GpuArray<T> source, {
  required List<int> resultShape,
  GpuArray<T>? out,
}) {
  validateLinalgOut(out, source.device, resultShape, source.dtype);
  final axisLength = source.shape.last;
  var outerSize = 1;
  for (final dim in resultShape) {
    outerSize *= dim;
  }
  return ResourceScope.scope(() {
    if (outerSize == 0) {
      final emptyOut =
          out ??
          GpuArray<T>.empty(resultShape, source.dtype, device: source.device);
      if (out == null) emptyOut.detachToParentScope();
      return emptyOut;
    }
    if (isComplexDType(source.dtype)) {
      if (isSinglePrecisionDType(source.dtype)) {
        final inBuffer = toContiguousComplex64Buffer(source);
        final outBuffer = source.device.createBuffer(
          sizeInBytes: math.max(outerSize * 8, 8),
        );
        final module = getOrCreateLinalgShader(
          'linalg_sum_last_c64',
          () => _sumLastAxisC64Shader,
          workgroupSize: 64,
        );
        source.device.backend.dispatchComputePipeline(
          shaderModule: module,
          buffers: [inBuffer, outBuffer],
          uniforms: <int>[outerSize, axisLength, 0, 0],
          workgroupsX: (outerSize + 63) ~/ 64,
        );
        final result = writeComplex64BufferToArray<T>(
          source.device,
          outBuffer,
          resultShape,
          source.dtype,
          out: out,
        );
        if (out == null) result.detachToParentScope();
        return result;
      }
      final inBuffer = toContiguousComplex128Buffer(source);
      final outBuffer = source.device.createBuffer(
        sizeInBytes: math.max(outerSize * 16, 16),
      );
      final module = getOrCreateLinalgShader(
        'linalg_sum_last_c128',
        () => _sumLastAxisC128Shader,
        workgroupSize: 64,
      );
      source.device.backend.dispatchComputePipeline(
        shaderModule: module,
        buffers: [inBuffer, outBuffer],
        uniforms: <int>[outerSize, axisLength, 0, 0],
        workgroupsX: (outerSize + 63) ~/ 64,
      );
      final result = writeComplex128BufferToArray<T>(
        source.device,
        outBuffer,
        resultShape,
        source.dtype,
        out: out,
      );
      if (out == null) result.detachToParentScope();
      return result;
    } else {
      if (isSinglePrecisionDType(source.dtype)) {
        final inBuffer = toContiguousFloat32Buffer(source);
        final outBuffer = source.device.createBuffer(
          sizeInBytes: math.max(outerSize * 4, 4),
        );
        final module = getOrCreateLinalgShader(
          'linalg_sum_last_f32',
          () => _sumLastAxisF32Shader,
          workgroupSize: 64,
        );
        source.device.backend.dispatchComputePipeline(
          shaderModule: module,
          buffers: [inBuffer, outBuffer],
          uniforms: <int>[outerSize, axisLength, 0, 0],
          workgroupsX: (outerSize + 63) ~/ 64,
        );
        final result = writeFloat32BufferToArray<T>(
          source.device,
          outBuffer,
          resultShape,
          source.dtype,
          out: out,
        );
        if (out == null) result.detachToParentScope();
        return result;
      }
      final inBuffer = toContiguousFloat64Buffer(source);
      final outBuffer = source.device.createBuffer(
        sizeInBytes: math.max(outerSize * 8, 8),
      );
      final module = getOrCreateLinalgShader(
        'linalg_sum_last_f64',
        () => _sumLastAxisF64Shader,
        workgroupSize: 64,
      );
      source.device.backend.dispatchComputePipeline(
        shaderModule: module,
        buffers: [inBuffer, outBuffer],
        uniforms: <int>[outerSize, axisLength, 0, 0],
        workgroupsX: (outerSize + 63) ~/ 64,
      );
      final result = writeFloat64BufferToArray<T>(
        source.device,
        outBuffer,
        resultShape,
        source.dtype,
        out: out,
      );
      if (out == null) result.detachToParentScope();
      return result;
    }
  });
}
