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

import '../../dtype.dart';
import 'wgsl_dtype_codec.dart';
import 'wgsl_types.dart';

/// Universal 15-[DType] WGSL compute shader generators for arithmetic, casts,
/// reductions, matrix multiplication, fills, and sequence generation.
extension type const WgslUniversalKernels._(Object? _) {
  static final Map<(String, DType, bool), WgslShaderModule> _binaryCache = {};
  static final Map<(String, DType, bool), WgslShaderModule> _unaryCache = {};
  static final Map<(DType, DType), WgslShaderModule> _castOrCopyCache = {};
  static final Map<(String, DType, DType), WgslShaderModule>
  _complexComponentCache = {};
  static final Map<DType, WgslShaderModule> _nanToNumCache = {};
  static final Map<DType, WgslShaderModule> _isCloseCache = {};
  static final Map<(String, DType, DType), WgslShaderModule> _reductionCache =
      {};
  static final Map<DType, WgslShaderModule> _matmulCache = {};
  static final Map<DType, WgslShaderModule> _fillCache = {};
  static final Map<DType, WgslShaderModule> _generatorCache = {};

  /// Standard strided metadata struct and offset helpers in WGSL.
  static const String stridedMetadataWgsl = '''
struct StridedMetadata {
  total_elements: u32,
  rank: u32,
  pad0: u32,
  pad1: u32,
  shape: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_b: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
  offset_a: u32,
  offset_b: u32,
  offset_out: u32,
  scalar_param: f32,
};

fn get_shape_dim(params: StridedMetadata, d: u32) -> u32 {
  return params.shape[d / 4u][d % 4u];
}

fn get_stride_a(params: StridedMetadata, d: u32) -> i32 {
  return params.strides_a[d / 4u][d % 4u];
}

fn get_stride_b(params: StridedMetadata, d: u32) -> i32 {
  return params.strides_b[d / 4u][d % 4u];
}

fn get_stride_out(params: StridedMetadata, d: u32) -> i32 {
  return params.strides_out[d / 4u][d % 4u];
}

fn flat_to_strided_offsets(
  flat_idx: u32,
  params: StridedMetadata,
  off_a: ptr<function, u32>,
  off_b: ptr<function, u32>,
  off_out: ptr<function, u32>
) {
  if ((params.pad0 & 1u) == 1u) {
    *off_a = params.offset_a + flat_idx;
    *off_b = params.offset_b + flat_idx;
    *off_out = params.offset_out + flat_idx;
    return;
  }
  var rem = flat_idx;
  var a_idx: i32 = i32(params.offset_a);
  var b_idx: i32 = i32(params.offset_b);
  var out_idx: i32 = i32(params.offset_out);

  for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
    let d = params.rank - 1u - i;
    let dim_size = get_shape_dim(params, d);
    let coord = i32(rem % dim_size);
    rem = rem / dim_size;
    a_idx = a_idx + coord * get_stride_a(params, d);
    b_idx = b_idx + coord * get_stride_b(params, d);
    out_idx = out_idx + coord * get_stride_out(params, d);
  }

  *off_a = u32(a_idx);
  *off_b = u32(b_idx);
  *off_out = u32(out_idx);
}
''';

  /// Generates a strided elementwise binary or comparison compute shader for
  /// [dtype].
  static WgslShaderModule binaryShader({
    required String op,
    required DType dtype,
    bool isComparison = false,
  }) {
    final cacheKey = (op, dtype, isComparison);
    if (_binaryCache[cacheKey] case final cached?) {
      return cached;
    }
    final outDType = isComparison ? DType.boolean : dtype;
    final expr = isComparison
        ? 'select(0u, 1u, ${WgslDTypeCodec.comparisonBoolExpr(op, dtype)})'
        : WgslDTypeCodec.binaryValueExpr(op, dtype);

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}
$stridedMetadataWgsl

${WgslDTypeCodec.readBindingDecl(0, 'src_a', dtype)}
${WgslDTypeCodec.readBindingDecl(1, 'src_b', dtype)}
${WgslDTypeCodec.writeBindingDecl(2, 'dst', outDType)}
@group(0) @binding(3) var<uniform> metadata: StridedMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_a', 'src_a', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_b', 'src_b', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', outDType)}
${WgslDTypeCodec.computeLoadFunction('load_val_a', 'load_raw_a', dtype)}
${WgslDTypeCodec.computeLoadFunction('load_val_b', 'load_raw_b', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', outDType)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }
  var off_a: u32 = 0u;
  var off_b: u32 = 0u;
  var off_dst: u32 = 0u;
  flat_to_strided_offsets(idx, metadata, &off_a, &off_b, &off_dst);
  let a = load_val_a(off_a);
  let b = load_val_b(off_b);
  store_val_dst(off_dst, $expr);
}
''';

    return _binaryCache[cacheKey] = WgslShaderModule(
      name: 'elementwise_binary_${op}_${dtype.name}_strided',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src_a',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'src_b',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst',
          dtype: WgslDType.fromDType(outDType),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'StridedMetadata',
        ),
      ],
    );
  }

  /// Generates a strided elementwise unary or predicate compute shader for
  /// [dtype].
  static WgslShaderModule unaryShader({
    required String op,
    required DType dtype,
    bool isPredicate = false,
  }) {
    final effectivePred =
        isPredicate ||
        op == 'isnan' ||
        op == 'isinf' ||
        op == 'isfinite' ||
        op == 'signbit';
    final cacheKey = (op, dtype, effectivePred);
    if (_unaryCache[cacheKey] case final cached?) {
      return cached;
    }
    final outDType = effectivePred ? DType.boolean : dtype;
    final expr = effectivePred
        ? 'select(0u, 1u, ${WgslDTypeCodec.unaryPredicateBoolExpr(op, dtype)})'
        : WgslDTypeCodec.unaryValueExpr(op, dtype);
    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}
$stridedMetadataWgsl

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', outDType)}
@group(0) @binding(2) var<uniform> metadata: StridedMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', outDType)}
${WgslDTypeCodec.computeLoadFunction('load_val_src', 'load_raw_src', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', outDType)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }
  var off_src: u32 = 0u;
  var dummy_b: u32 = 0u;
  var off_dst: u32 = 0u;
  flat_to_strided_offsets(idx, metadata, &off_src, &dummy_b, &off_dst);
  let a = load_val_src(off_src);
  store_val_dst(off_dst, $expr);
}
''';

    return _unaryCache[cacheKey] = WgslShaderModule(
      name: 'elementwise_unary_${op}_${dtype.name}_strided',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: WgslDType.fromDType(outDType),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'StridedMetadata',
        ),
      ],
    );
  }

  /// Generates a strided complex component or phase extraction shader (`real`,
  /// `imag`, `angle`) from [sourceDType] to [targetDType].
  static WgslShaderModule complexComponentShader({
    required String op,
    required DType sourceDType,
    required DType targetDType,
  }) {
    final cacheKey = (op, sourceDType, targetDType);
    if (_complexComponentCache[cacheKey] case final cached?) {
      return cached;
    }
    final extractReIm = switch (sourceDType) {
      DType.complex64 || DType.complex128 =>
        '''
  let s_val = load_val_src(off_a);
  let re = s_val.x;
  let im = s_val.y;''',
      DType.int64 =>
        '''
  let s_val = load_val_src(off_a);
  let re = i64_to_f32(s_val);
  let im = 0.0;''',
      DType.uint64 =>
        '''
  let s_val = load_val_src(off_a);
  let re = u64_to_f32(s_val);
  let im = 0.0;''',
      DType.int32 ||
      DType.int16 ||
      DType.int8 ||
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean =>
        '''
  let s_val = load_val_src(off_a);
  let re = f32(s_val);
  let im = 0.0;''',
      DType.float64 || DType.float32 || DType.float16 || DType.bfloat16 =>
        '''
  let s_val = load_val_src(off_a);
  let re = s_val;
  let im = 0.0;''',
    };

    String convertFromF32(String expr) => switch (targetDType) {
      DType.float64 || DType.float32 || DType.float16 || DType.bfloat16 => expr,
      DType.int64 => 'f32_to_i64(trunc($expr))',
      DType.uint64 => 'f32_to_u64(max(0.0, trunc($expr)))',
      DType.int32 || DType.int16 || DType.int8 => 'i32($expr)',
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 => 'u32(max(0.0, trunc($expr)))',
      DType.boolean => 'select(0u, 1u, ($expr) != 0.0)',
      DType.complex64 || DType.complex128 => 'vec2<f32>($expr, 0.0)',
    };

    final String body;
    if (op == 'real') {
      if (sourceDType == DType.complex128 && targetDType == DType.float64) {
        body = 'let raw = load_raw_src(off_a); store_raw_dst(off_out, raw.xy);';
      } else if (sourceDType == targetDType && !sourceDType.isComplex) {
        body = 'store_raw_dst(off_out, load_raw_src(off_a));';
      } else {
        body =
            '''
$extractReIm
  store_val_dst(off_out, ${convertFromF32('re')});''';
      }
    } else if (op == 'imag') {
      if (sourceDType == DType.complex128 && targetDType == DType.float64) {
        body = 'let raw = load_raw_src(off_a); store_raw_dst(off_out, raw.zw);';
      } else if (!sourceDType.isComplex) {
        body =
            'store_raw_dst(off_out, ${WgslDTypeCodec.rawZeroLiteral(targetDType)});';
      } else {
        body =
            '''
$extractReIm
  store_val_dst(off_out, ${convertFromF32('im')});''';
      }
    } else {
      body =
          '''
$extractReIm
  let ang = atan2(im, re) * metadata.scalar_param;
  store_val_dst(off_out, ${convertFromF32('ang')});''';
    }

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}
$stridedMetadataWgsl

${WgslDTypeCodec.readBindingDecl(0, 'src', sourceDType)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', targetDType)}
@group(0) @binding(2) var<uniform> metadata: StridedMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', sourceDType)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', targetDType)}
${WgslDTypeCodec.computeLoadFunction('load_val_src', 'load_raw_src', sourceDType)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', targetDType)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }
  var off_a: u32 = 0u;
  var dummy_b: u32 = 0u;
  var off_out: u32 = 0u;
  flat_to_strided_offsets(idx, metadata, &off_a, &dummy_b, &off_out);
  $body
}
''';

    return _complexComponentCache[cacheKey] = WgslShaderModule(
      name: 'complex_${op}_${sourceDType.name}_to_${targetDType.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: WgslDType.fromDType(sourceDType),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: WgslDType.fromDType(targetDType),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'StridedMetadata',
        ),
      ],
    );
  }

  /// Generates a strided `nanToNum` compute shader for [dtype] across all 15
  /// [DType]s.
  static WgslShaderModule nanToNumShader(DType dtype) {
    if (_nanToNumCache[dtype] case final cached?) {
      return cached;
    }
    final body = switch (dtype) {
      DType.float64 =>
        'store_raw_dst(off_out, replace_f64(load_raw_src(off_a)));',
      DType.complex128 =>
        '''
  let raw = load_raw_src(off_a);
  let re = replace_f64(raw.xy);
  let im = replace_f64(raw.zw);
  store_raw_dst(off_out, vec4<u32>(re.x, re.y, im.x, im.y));''',
      DType.float32 || DType.float16 || DType.bfloat16 =>
        'store_val_dst(off_out, replace_f32(load_val_src(off_a)));',
      DType.complex64 =>
        '''
  let val = load_val_src(off_a);
  store_val_dst(off_out, vec2<f32>(replace_f32(val.x), replace_f32(val.y)));''',
      DType.int64 ||
      DType.uint64 ||
      DType.int32 ||
      DType.int16 ||
      DType.int8 ||
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean => 'store_raw_dst(off_out, load_raw_src(off_a));',
    };

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}

struct NanToNumUniforms {
  total_elements: u32,
  rank: u32,
  pad0: u32,
  pad1: u32,
  shape: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
  offset_a: u32,
  offset_out: u32,
  nan_lo: u32,
  nan_hi: u32,
  posinf_lo: u32,
  posinf_hi: u32,
  neginf_lo: u32,
  neginf_hi: u32,
  nan_f32: f32,
  posinf_f32: f32,
  neginf_f32: f32,
  pad2: u32,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> uniforms: NanToNumUniforms;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}
${WgslDTypeCodec.computeLoadFunction('load_val_src', 'load_raw_src', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', dtype)}

fn replace_f64(raw: vec2<u32>) -> vec2<u32> {
  let exp64 = (raw.y >> 20u) & 0x7FFu;
  let mant_nz = ((raw.y & 0xFFFFFu) != 0u) || (raw.x != 0u);
  if (exp64 == 0x7FFu) {
    if (mant_nz) {
      return vec2<u32>(uniforms.nan_lo, uniforms.nan_hi);
    }
    if ((raw.y & 0x80000000u) != 0u) {
      return vec2<u32>(uniforms.neginf_lo, uniforms.neginf_hi);
    }
    return vec2<u32>(uniforms.posinf_lo, uniforms.posinf_hi);
  }
  return raw;
}

fn replace_f32(v: f32) -> f32 {
  if (f32_isnan(v)) {
    return uniforms.nan_f32;
  }
  if (f32_isinf(v)) {
    return select(uniforms.posinf_f32, uniforms.neginf_f32, f32_signbit(v));
  }
  return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= uniforms.total_elements) {
    return;
  }
  var off_a: u32 = uniforms.offset_a + idx;
  var off_out: u32 = uniforms.offset_out + idx;
  if (uniforms.pad0 != 1u) {
    var rem = idx;
    var a_idx: i32 = i32(uniforms.offset_a);
    var out_idx: i32 = i32(uniforms.offset_out);
    for (var i: u32 = 0u; i < uniforms.rank; i = i + 1u) {
      let d = uniforms.rank - 1u - i;
      let dim_size = uniforms.shape[d / 4u][d % 4u];
      let coord = i32(rem % dim_size);
      rem = rem / dim_size;
      a_idx = a_idx + coord * uniforms.strides_a[d / 4u][d % 4u];
      out_idx = out_idx + coord * uniforms.strides_out[d / 4u][d % 4u];
    }
    off_a = u32(a_idx);
    off_out = u32(out_idx);
  }
  $body
}
''';

    return _nanToNumCache[dtype] = WgslShaderModule(
      name: 'nan_to_num_${dtype.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'NanToNumUniforms',
        ),
      ],
    );
  }

  /// Generates a strided `isClose` elementwise tolerance comparison shader for
  /// [dtype] outputting [DType.boolean].
  static WgslShaderModule isCloseShader(DType dtype) {
    if (_isCloseCache[dtype] case final cached?) {
      return cached;
    }
    final checkBody = switch (dtype) {
      DType.float64 || DType.float32 || DType.float16 || DType.bfloat16 =>
        '''
  let rtol = bitcast<f32>(metadata.pad1);
  let atol = metadata.scalar_param;
  let eq_nan = (metadata.pad0 & 2u) != 0u;
  let nan_a = f32_isnan(a);
  let nan_b = f32_isnan(b);
  var close: bool = false;
  if (nan_a || nan_b) {
    close = eq_nan && nan_a && nan_b;
  } else if (f32_isinf(a) || f32_isinf(b)) {
    close = a == b;
  } else {
    close = abs(a - b) <= (atol + rtol * abs(b));
  }''',
      DType.complex64 || DType.complex128 =>
        '''
  let rtol = bitcast<f32>(metadata.pad1);
  let atol = metadata.scalar_param;
  let eq_nan = (metadata.pad0 & 2u) != 0u;
  let nan_a = f32_isnan(a.x) || f32_isnan(a.y);
  let nan_b = f32_isnan(b.x) || f32_isnan(b.y);
  var close: bool = false;
  if (nan_a || nan_b) {
    close = eq_nan && nan_a && nan_b;
  } else if (f32_isinf(a.x) || f32_isinf(a.y) || f32_isinf(b.x) || f32_isinf(b.y)) {
    close = (a.x == b.x) && (a.y == b.y);
  } else {
    close = length(a - b) <= (atol + rtol * length(b));
  }''',
      DType.int64 =>
        '''
  let fa = i64_to_f32(a);
  let fb = i64_to_f32(b);
  let rtol = bitcast<f32>(metadata.pad1);
  let atol = metadata.scalar_param;
  let close = abs(fa - fb) <= (atol + rtol * abs(fb));''',
      DType.uint64 =>
        '''
  let fa = u64_to_f32(a);
  let fb = u64_to_f32(b);
  let rtol = bitcast<f32>(metadata.pad1);
  let atol = metadata.scalar_param;
  let close = abs(fa - fb) <= (atol + rtol * abs(fb));''',
      DType.int32 ||
      DType.int16 ||
      DType.int8 ||
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean =>
        '''
  let fa = f32(a);
  let fb = f32(b);
  let rtol = bitcast<f32>(metadata.pad1);
  let atol = metadata.scalar_param;
  let close = abs(fa - fb) <= (atol + rtol * abs(fb));''',
    };

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}
$stridedMetadataWgsl

${WgslDTypeCodec.readBindingDecl(0, 'src_a', dtype)}
${WgslDTypeCodec.readBindingDecl(1, 'src_b', dtype)}
${WgslDTypeCodec.writeBindingDecl(2, 'dst', DType.boolean)}
@group(0) @binding(3) var<uniform> metadata: StridedMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_a', 'src_a', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_b', 'src_b', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', DType.boolean)}
${WgslDTypeCodec.computeLoadFunction('load_val_a', 'load_raw_a', dtype)}
${WgslDTypeCodec.computeLoadFunction('load_val_b', 'load_raw_b', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', DType.boolean)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }
  var off_a: u32 = 0u;
  var off_b: u32 = 0u;
  var off_dst: u32 = 0u;
  flat_to_strided_offsets(idx, metadata, &off_a, &off_b, &off_dst);
  let a = load_val_a(off_a);
  let b = load_val_b(off_b);
$checkBody
  store_val_dst(off_dst, select(0u, 1u, close));
}
''';

    return _isCloseCache[dtype] = WgslShaderModule(
      name: 'is_close_${dtype.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src_a',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'src_b',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst',
          dtype: WgslDType.fromDType(DType.boolean),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'StridedMetadata',
        ),
      ],
    );
  }

  /// Generates a strided copy or DType cast compute shader from [sourceDType]
  /// to [targetDType].
  static WgslShaderModule castOrCopyShader({
    required DType sourceDType,
    required DType targetDType,
  }) {
    final cacheKey = (sourceDType, targetDType);
    if (_castOrCopyCache[cacheKey] case final cached?) {
      return cached;
    }
    final sameType = sourceDType == targetDType;
    final String body;
    if (sameType) {
      body = 'store_raw_dst(off_out, load_raw_src(off_a));';
    } else {
      final extractExpr = switch (sourceDType) {
        DType.complex64 || DType.complex128 =>
          '''
  let s_val = load_val_src(off_a);
  let re = s_val.x;
  let im = s_val.y;
  let is_nz = (re != 0.0) || (im != 0.0);
  let i64_val = f32_to_i64(trunc(re));''',
        DType.int64 =>
          '''
  let s_val = load_val_src(off_a);
  let re = i64_to_f32(s_val);
  let im = 0.0;
  let is_nz = (s_val.x != 0u) || (s_val.y != 0u);
  let i64_val = s_val;''',
        DType.uint64 =>
          '''
  let s_val = load_val_src(off_a);
  let re = u64_to_f32(s_val);
  let im = 0.0;
  let is_nz = (s_val.x != 0u) || (s_val.y != 0u);
  let i64_val = s_val;''',
        DType.int32 || DType.int16 || DType.int8 =>
          '''
  let s_val = load_val_src(off_a);
  let re = f32(s_val);
  let im = 0.0;
  let is_nz = s_val != 0;
  let i64_val = vec2<u32>(bitcast<u32>(s_val), select(0u, 0xFFFFFFFFu, s_val < 0));''',
        DType.uint32 || DType.uint16 || DType.uint8 || DType.boolean =>
          '''
  let s_val = load_val_src(off_a);
  let re = f32(s_val);
  let im = 0.0;
  let is_nz = s_val != 0u;
  let i64_val = vec2<u32>(s_val, 0u);''',
        DType.float64 || DType.float32 || DType.float16 || DType.bfloat16 =>
          '''
  let s_val = load_val_src(off_a);
  let re = s_val;
  let im = 0.0;
  let is_nz = s_val != 0.0;
  let i64_val = f32_to_i64(trunc(re));''',
      };

      final convertExpr = switch (targetDType) {
        DType.complex64 || DType.complex128 => 'vec2<f32>(re, im)',
        DType.boolean => 'select(0u, 1u, is_nz)',
        DType.int64 || DType.uint64 => 'i64_val',
        DType.int32 || DType.int16 || DType.int8 =>
          sourceDType.isInteger || sourceDType == DType.boolean
              ? 'bitcast<i32>(i64_val.x)'
              : 'i32(re)',
        DType.uint32 || DType.uint16 || DType.uint8 =>
          sourceDType.isInteger || sourceDType == DType.boolean
              ? 'i64_val.x'
              : 'u32(max(0.0, trunc(re)))',
        DType.float64 ||
        DType.float32 ||
        DType.float16 ||
        DType.bfloat16 => 're',
      };

      body =
          '''
$extractExpr
  store_val_dst(off_out, $convertExpr);''';
    }

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}
$stridedMetadataWgsl

${WgslDTypeCodec.readBindingDecl(0, 'src', sourceDType)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', targetDType)}
@group(0) @binding(2) var<uniform> metadata: StridedMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', sourceDType)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', targetDType)}
${WgslDTypeCodec.computeLoadFunction('load_val_src', 'load_raw_src', sourceDType)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', targetDType)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }
  var off_a: u32 = 0u;
  var dummy_b: u32 = 0u;
  var off_out: u32 = 0u;
  flat_to_strided_offsets(idx, metadata, &off_a, &dummy_b, &off_out);
  $body
}
''';

    return _castOrCopyCache[cacheKey] = WgslShaderModule(
      name: 'copy_cast_${sourceDType.name}_to_${targetDType.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: WgslDType.fromDType(sourceDType),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: WgslDType.fromDType(targetDType),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'StridedMetadata',
        ),
      ],
    );
  }

  /// Generates a universal strided reduction compute shader (supporting full
  /// and single-axis reductions for `sum`, `mean`, `prod`, `min`, `max`,
  /// `argmin`, `argmax`, `all`, `any`, `count_nonzero`, `ptp`, `nansum`,
  /// `nanmean`, `nanmin`, `nanmax`, `variance`, `std`) across all 15 [DType]s.
  static WgslShaderModule reductionShader({
    required String op,
    required DType dtype,
    DType? targetDType,
  }) {
    final isArgOp = op == 'argmin' || op == 'argmax';
    final isCountNonzero = op == 'count_nonzero' || op == 'countNonzero';
    final isBoolOp = op == 'all' || op == 'any';
    final outDType =
        targetDType ??
        ((isArgOp || isCountNonzero)
            ? DType.int64
            : (isBoolOp ? DType.boolean : dtype));
    final cacheKey = (op, dtype, outDType);
    if (_reductionCache[cacheKey] case final cached?) {
      return cached;
    }
    final valType = WgslDTypeCodec.computeValueType(dtype);

    final zeroExpr = switch (dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => '0.0',
      DType.int32 || DType.int16 || DType.int8 => '0',
      DType.uint32 || DType.uint16 || DType.uint8 || DType.boolean => '0u',
      DType.int64 || DType.uint64 => 'vec2<u32>(0u, 0u)',
      DType.complex64 || DType.complex128 => 'vec2<f32>(0.0, 0.0)',
    };
    final oneExpr = switch (dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => '1.0',
      DType.int32 || DType.int16 || DType.int8 => '1',
      DType.uint32 || DType.uint16 || DType.uint8 || DType.boolean => '1u',
      DType.int64 || DType.uint64 => 'vec2<u32>(1u, 0u)',
      DType.complex64 || DType.complex128 => 'vec2<f32>(1.0, 0.0)',
    };
    final nonZeroExpr = switch (dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => 'val != 0.0',
      DType.int32 || DType.int16 || DType.int8 => 'val != 0',
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean => 'val != 0u',
      DType.int64 || DType.uint64 => '(val.x != 0u) || (val.y != 0u)',
      DType.complex64 || DType.complex128 => '(val.x != 0.0) || (val.y != 0.0)',
    };
    final addExpr = WgslDTypeCodec.binaryValueExpr('add', dtype);
    final subExpr = WgslDTypeCodec.binaryValueExpr('sub', dtype);
    final mulExpr = WgslDTypeCodec.binaryValueExpr('mul', dtype);
    final minExpr = WgslDTypeCodec.binaryValueExpr('min', dtype);
    final maxExpr = WgslDTypeCodec.binaryValueExpr('max', dtype);
    final meanFinishExpr = switch (dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => 'acc / f32(metadata.axis_size)',
      DType.int32 ||
      DType.int16 ||
      DType.int8 => 'acc / i32(metadata.axis_size)',
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean => 'acc / metadata.axis_size',
      DType.int64 => 'f32_to_i64(i64_to_f32(acc) / f32(metadata.axis_size))',
      DType.uint64 => 'f32_to_u64(u64_to_f32(acc) / f32(metadata.axis_size))',
      DType.complex64 || DType.complex128 =>
        'vec2<f32>(acc.x / f32(metadata.axis_size), acc.y / f32(metadata.axis_size))',
    };

    final String sharedDecls;
    final String parallelReductionBody;
    final String reductionLoopBody;
    if (isArgOp) {
      final cmpCond = WgslDTypeCodec.comparisonBoolExpr(
        op == 'argmin' ? 'lt' : 'gt',
        dtype,
      );
      final storeBestExpr = switch (outDType) {
        DType.int64 ||
        DType.uint64 => 'store_raw_dst(u32(off_out), vec2<u32>(best_idx, 0u));',
        _ => 'store_raw_dst(u32(off_out), best_idx);',
      };
      sharedDecls =
          '''
var<workgroup> shared_val: array<$valType, 256>;
var<workgroup> shared_idx: array<u32, 256>;''';
      parallelReductionBody =
          '''
  if (metadata.total_out == 1u && metadata.axis_size >= 256u) {
    let tid = local_id.x;
    let base_off_a: i32 = i32(metadata.offset_a);
    let off_out: i32 = i32(metadata.offset_out);
    let axis_stride = select(0, get_stride_a(metadata, metadata.reduce_axis), metadata.reduce_axis < metadata.rank);
    var best_val: $valType = load_val_src(step_off_a(tid, base_off_a, axis_stride, metadata));
    var best_idx: u32 = tid;
    for (var step: u32 = tid + 256u; step < metadata.axis_size; step = step + 256u) {
      let a = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
      let b = best_val;
      if ($cmpCond) {
        best_val = a;
        best_idx = step;
      }
    }
    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        let cand_val = shared_val[tid + s];
        let cand_idx = shared_idx[tid + s];
        let curr_val = shared_val[tid];
        let curr_idx = shared_idx[tid];
        var a = cand_val;
        var b = curr_val;
        let cand_better = $cmpCond;
        a = curr_val;
        b = cand_val;
        let curr_better = $cmpCond;
        if (cand_better || (!curr_better && cand_idx < curr_idx)) {
          shared_val[tid] = cand_val;
          shared_idx[tid] = cand_idx;
        }
      }
      workgroupBarrier();
    }
    if (tid == 0u) {
      let best_idx = shared_idx[0u];
      $storeBestExpr
    }
    return;
  }''';
      reductionLoopBody =
          '''
  var best_val: $valType = load_val_src(step_off_a(0u, base_off_a, axis_stride, metadata));
  var best_idx: u32 = 0u;
  for (var step: u32 = 1u; step < metadata.axis_size; step = step + 1u) {
    let a = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    let b = best_val;
    if ($cmpCond) {
      best_val = a;
      best_idx = step;
    }
  }
  $storeBestExpr''';
    } else if (isCountNonzero) {
      final storeCountExpr = switch (outDType) {
        DType.int64 || DType.uint64 =>
          'store_raw_dst(u32(off_out), vec2<u32>(acc_count, 0u));',
        _ => 'store_raw_dst(u32(off_out), acc_count);',
      };
      sharedDecls = 'var<workgroup> shared_count: array<u32, 256>;';
      parallelReductionBody =
          '''
  if (metadata.total_out == 1u && metadata.axis_size >= 256u) {
    let tid = local_id.x;
    let base_off_a: i32 = i32(metadata.offset_a);
    let off_out: i32 = i32(metadata.offset_out);
    let axis_stride = select(0, get_stride_a(metadata, metadata.reduce_axis), metadata.reduce_axis < metadata.rank);
    var acc_count: u32 = 0u;
    for (var step: u32 = tid; step < metadata.axis_size; step = step + 256u) {
      let val = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
      if ($nonZeroExpr) {
        acc_count = acc_count + 1u;
      }
    }
    shared_count[tid] = acc_count;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        shared_count[tid] = shared_count[tid] + shared_count[tid + s];
      }
      workgroupBarrier();
    }
    if (tid == 0u) {
      let acc_count = shared_count[0u];
      $storeCountExpr
    }
    return;
  }''';
      reductionLoopBody =
          '''
  var acc_count: u32 = 0u;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let val = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    if ($nonZeroExpr) {
      acc_count = acc_count + 1u;
    }
  }
  $storeCountExpr''';
    } else if (isBoolOp) {
      final initBool = op == 'all' ? 'true' : 'false';
      final combineBool = op == 'all' ? 'acc && is_nz' : 'acc || is_nz';
      sharedDecls = 'var<workgroup> shared_bool: array<u32, 256>;';
      parallelReductionBody =
          '''
  if (metadata.total_out == 1u && metadata.axis_size >= 256u) {
    let tid = local_id.x;
    let base_off_a: i32 = i32(metadata.offset_a);
    let off_out: i32 = i32(metadata.offset_out);
    let axis_stride = select(0, get_stride_a(metadata, metadata.reduce_axis), metadata.reduce_axis < metadata.rank);
    var acc: bool = $initBool;
    for (var step: u32 = tid; step < metadata.axis_size; step = step + 256u) {
      let val = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
      let is_nz = $nonZeroExpr;
      acc = $combineBool;
    }
    shared_bool[tid] = select(0u, 1u, acc);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        let lhs = shared_bool[tid] != 0u;
        let rhs = shared_bool[tid + s] != 0u;
        shared_bool[tid] = select(0u, 1u, ${op == 'all' ? 'lhs && rhs' : 'lhs || rhs'});
      }
      workgroupBarrier();
    }
    if (tid == 0u) {
      store_val_dst(u32(off_out), shared_bool[0u]);
    }
    return;
  }''';
      reductionLoopBody =
          '''
  var acc: bool = $initBool;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let val = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    let is_nz = $nonZeroExpr;
    acc = $combineBool;
  }
  store_val_dst(u32(off_out), select(0u, 1u, acc));''';
    } else if (op == 'ptp') {
      sharedDecls = '';
      parallelReductionBody = '';
      final ptpExpr = dtype == DType.boolean
          ? 'select(0u, 1u, max_acc != min_acc)'
          : subExpr;
      reductionLoopBody =
          '''
  var min_acc: $valType = load_val_src(step_off_a(0u, base_off_a, axis_stride, metadata));
  var max_acc: $valType = min_acc;
  for (var step: u32 = 1u; step < metadata.axis_size; step = step + 1u) {
    let cur = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    {
      let a = min_acc;
      let b = cur;
      min_acc = $minExpr;
    }
    {
      let a = max_acc;
      let b = cur;
      max_acc = $maxExpr;
    }
  }
  let a = max_acc;
  let b = min_acc;
  store_val_dst(u32(off_out), $ptpExpr);''';
    } else if (op == 'variance' || op == 'std') {
      sharedDecls = '';
      parallelReductionBody = '';
      if (dtype.isComplex) {
        reductionLoopBody = '''
  var mean_acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    mean_acc = mean_acc + load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
  }
  let mean_val = mean_acc / f32(metadata.axis_size);
  var sq_acc: f32 = 0.0;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let diff = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata)) - mean_val;
    sq_acc = sq_acc + diff.x * diff.x + diff.y * diff.y;
  }
  let denom = f32(i32(metadata.axis_size) - i32(metadata.pad1));
  let var_val = select(f32_nan(), sq_acc / denom, denom > 0.0);
  store_val_dst(u32(off_out), ${op == 'std' ? 'sqrt(var_val)' : 'var_val'});''';
      } else {
        reductionLoopBody = '''
  var mean_acc: f32 = 0.0;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    mean_acc = mean_acc + load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
  }
  let mean_val = mean_acc / f32(metadata.axis_size);
  var sq_acc: f32 = 0.0;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let diff = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata)) - mean_val;
    sq_acc = sq_acc + diff * diff;
  }
  let denom = f32(i32(metadata.axis_size) - i32(metadata.pad1));
  let var_val = select(f32_nan(), sq_acc / denom, denom > 0.0);
  store_val_dst(u32(off_out), ${op == 'std' ? 'sqrt(var_val)' : 'var_val'});''';
      }
    } else if ((op == 'nanmin' || op == 'nanmax') &&
        (dtype.isFloating || dtype.isComplex)) {
      sharedDecls = '';
      parallelReductionBody = '';
      if (dtype.isComplex) {
        final cpxCmp = op == 'nanmin' ? minExpr : maxExpr;
        reductionLoopBody =
            '''
  var acc: vec2<f32> = vec2<f32>(f32_nan(), f32_nan());
  var has_valid: bool = false;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    if (!f32_isnan(b.x) && !f32_isnan(b.y)) {
      if (!has_valid) {
        acc = b;
        has_valid = true;
      } else {
        let a = acc;
        acc = $cpxCmp;
      }
    }
  }
  store_val_dst(u32(off_out), acc);''';
      } else {
        final fltCmp = op == 'nanmin' ? 'min(acc, b)' : 'max(acc, b)';
        reductionLoopBody =
            '''
  var acc: f32 = f32_nan();
  var has_valid: bool = false;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    if (!f32_isnan(b)) {
      if (!has_valid) {
        acc = b;
        has_valid = true;
      } else {
        acc = $fltCmp;
      }
    }
  }
  store_val_dst(u32(off_out), acc);''';
      }
    } else if ((op == 'nansum' || op == 'nanmean') &&
        (dtype.isFloating || dtype.isComplex)) {
      sharedDecls = '';
      parallelReductionBody = '';
      if (dtype.isComplex) {
        final finishExpr = op == 'nanmean'
            ? 'select(vec2<f32>(f32_nan(), f32_nan()), acc / f32(valid_count), valid_count > 0u)'
            : 'acc';
        reductionLoopBody = '''
  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  var valid_count: u32 = 0u;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    if (!f32_isnan(b.x) && !f32_isnan(b.y)) {
      acc = acc + b;
      valid_count = valid_count + 1u;
    }
  }
  store_val_dst(u32(off_out), $finishExpr);''';
      } else {
        final finishExpr = op == 'nanmean'
            ? 'select(f32_nan(), acc / f32(valid_count), valid_count > 0u)'
            : 'acc';
        reductionLoopBody = '''
  var acc: f32 = 0.0;
  var valid_count: u32 = 0u;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    if (!f32_isnan(b)) {
      acc = acc + b;
      valid_count = valid_count + 1u;
    }
  }
  store_val_dst(u32(off_out), $finishExpr);''';
      }
    } else if (op == 'min' || op == 'max' || op == 'nanmin' || op == 'nanmax') {
      final isMin = op == 'min' || op == 'nanmin';
      final combineVal = isMin ? minExpr : maxExpr;
      sharedDecls = 'var<workgroup> shared_val: array<$valType, 256>;';
      parallelReductionBody =
          '''
  if (metadata.total_out == 1u && metadata.axis_size >= 256u) {
    let tid = local_id.x;
    let base_off_a: i32 = i32(metadata.offset_a);
    let off_out: i32 = i32(metadata.offset_out);
    let axis_stride = select(0, get_stride_a(metadata, metadata.reduce_axis), metadata.reduce_axis < metadata.rank);
    var acc: $valType = load_val_src(step_off_a(tid, base_off_a, axis_stride, metadata));
    for (var step: u32 = tid + 256u; step < metadata.axis_size; step = step + 256u) {
      let a = acc;
      let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
      acc = $combineVal;
    }
    shared_val[tid] = acc;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        let a = shared_val[tid];
        let b = shared_val[tid + s];
        shared_val[tid] = $combineVal;
      }
      workgroupBarrier();
    }
    if (tid == 0u) {
      store_val_dst(u32(off_out), shared_val[0u]);
    }
    return;
  }''';
      reductionLoopBody =
          '''
  var acc: $valType = load_val_src(step_off_a(0u, base_off_a, axis_stride, metadata));
  for (var step: u32 = 1u; step < metadata.axis_size; step = step + 1u) {
    let a = acc;
    let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    acc = $combineVal;
  }
  store_val_dst(u32(off_out), acc);''';
    } else {
      final isProd = op == 'prod' || op == 'product';
      final initVal = isProd ? oneExpr : zeroExpr;
      final combineVal = isProd ? mulExpr : addExpr;
      final finalVal = (op == 'mean' || op == 'nanmean')
          ? 'select($zeroExpr, $meanFinishExpr, metadata.axis_size > 0u)'
          : 'acc';
      sharedDecls = 'var<workgroup> shared_val: array<$valType, 256>;';
      parallelReductionBody =
          '''
  if (metadata.total_out == 1u && metadata.axis_size >= 256u) {
    let tid = local_id.x;
    let base_off_a: i32 = i32(metadata.offset_a);
    let off_out: i32 = i32(metadata.offset_out);
    let axis_stride = select(0, get_stride_a(metadata, metadata.reduce_axis), metadata.reduce_axis < metadata.rank);
    var acc: $valType = load_val_src(step_off_a(tid, base_off_a, axis_stride, metadata));
    for (var step: u32 = tid + 256u; step < metadata.axis_size; step = step + 256u) {
      let a = acc;
      let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
      acc = $combineVal;
    }
    shared_val[tid] = acc;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        let a = shared_val[tid];
        let b = shared_val[tid + s];
        shared_val[tid] = $combineVal;
      }
      workgroupBarrier();
    }
    if (tid == 0u) {
      let acc = shared_val[0u];
      store_val_dst(u32(off_out), $finalVal);
    }
    return;
  }''';
      reductionLoopBody =
          '''
  var acc: $valType = $initVal;
  for (var step: u32 = 0u; step < metadata.axis_size; step = step + 1u) {
    let a = acc;
    let b = load_val_src(step_off_a(step, base_off_a, axis_stride, metadata));
    acc = $combineVal;
  }
  store_val_dst(u32(off_out), $finalVal);''';
    }

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}

struct AxisReductionMetadata {
  total_out: u32,
  rank: u32,
  reduce_axis: u32,
  axis_size: u32,
  offset_a: u32,
  offset_out: u32,
  pad0: u32,
  pad1: u32,
  shape_a: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', outDType)}
@group(0) @binding(2) var<uniform> metadata: AxisReductionMetadata;

$sharedDecls

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', outDType)}
${WgslDTypeCodec.computeLoadFunction('load_val_src', 'load_raw_src', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', outDType)}

fn get_shape_a(params: AxisReductionMetadata, d: u32) -> u32 {
  return params.shape_a[d / 4u][d % 4u];
}

fn get_stride_a(params: AxisReductionMetadata, d: u32) -> i32 {
  return params.strides_a[d / 4u][d % 4u];
}

fn get_stride_out(params: AxisReductionMetadata, d: u32) -> i32 {
  return params.strides_out[d / 4u][d % 4u];
}

fn step_off_a(step: u32, base_off_a: i32, axis_stride: i32, params: AxisReductionMetadata) -> u32 {
  if (params.reduce_axis == 99u) {
    if (params.pad0 == 1u) {
      return params.offset_a + step;
    }
    var rem = step;
    var off: i32 = i32(params.offset_a);
    for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
      let d = params.rank - 1u - i;
      let dim_sz = get_shape_a(params, d);
      let coord = i32(rem % dim_sz);
      rem = rem / dim_sz;
      off = off + coord * get_stride_a(params, d);
    }
    return u32(off);
  }
  return u32(base_off_a + i32(step) * axis_stride);
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>
) {
$parallelReductionBody
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_out) {
    return;
  }

  var rem = idx;
  var base_off_a: i32 = i32(metadata.offset_a);
  var off_out: i32 = i32(metadata.offset_out);

  if (metadata.reduce_axis != 99u) {
    for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
      let d = metadata.rank - 1u - i;
      if (d == metadata.reduce_axis) {
        continue;
      }
      let out_d = select(d, d - 1u, d > metadata.reduce_axis);
      let dim_sz = get_shape_a(metadata, d);
      let coord = i32(rem % dim_sz);
      rem = rem / dim_sz;
      base_off_a = base_off_a + coord * get_stride_a(metadata, d);
      off_out = off_out + coord * get_stride_out(metadata, out_d);
    }
  }

  let axis_stride = select(0, get_stride_a(metadata, metadata.reduce_axis), metadata.reduce_axis < metadata.rank);
$reductionLoopBody
}
''';

    return _reductionCache[cacheKey] = WgslShaderModule(
      name: 'axis_reduction_${op}_${dtype.name}_${outDType.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: WgslDType.fromDType(outDType),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'AxisReductionMetadata',
        ),
      ],
    );
  }

  /// Generates a universal strided batched matrix multiplication shader for
  /// [dtype] across all 15 [DType]s.
  static WgslShaderModule matmulShader(DType dtype) {
    if (_matmulCache[dtype] case final cached?) {
      return cached;
    }
    final valType = WgslDTypeCodec.computeValueType(dtype);
    final zeroExpr = switch (dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => '0.0',
      DType.int32 || DType.int16 || DType.int8 => '0',
      DType.uint32 || DType.uint16 || DType.uint8 || DType.boolean => '0u',
      DType.int64 || DType.uint64 => 'vec2<u32>(0u, 0u)',
      DType.complex64 || DType.complex128 => 'vec2<f32>(0.0, 0.0)',
    };
    final mulExpr = WgslDTypeCodec.binaryValueExpr('mul', dtype);
    final addExpr = WgslDTypeCodec.binaryValueExpr(
      'add',
      dtype,
    ).replaceAll('a', 'acc').replaceAll('b', 'prod_val');

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}

struct MatmulMetadata {
  M: u32,
  N: u32,
  K: u32,
  batch_count: u32,
  batch_rank: u32,
  offset_a: u32,
  offset_b: u32,
  offset_out: u32,
  stride_a_m: i32,
  stride_a_k: i32,
  stride_b_k: i32,
  stride_b_n: i32,
  stride_out_m: i32,
  stride_out_n: i32,
  batch_offset: u32,
  pad0: u32,
  batch_shape: array<vec4<u32>, 2>,
  batch_strides_a: array<vec4<i32>, 2>,
  batch_strides_b: array<vec4<i32>, 2>,
  batch_strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'mat_a', dtype)}
${WgslDTypeCodec.readBindingDecl(1, 'mat_b', dtype)}
${WgslDTypeCodec.writeBindingDecl(2, 'mat_c', dtype)}
@group(0) @binding(3) var<uniform> metadata: MatmulMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_a', 'mat_a', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_b', 'mat_b', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_c', 'mat_c', dtype)}
${WgslDTypeCodec.computeLoadFunction('load_val_a', 'load_raw_a', dtype)}
${WgslDTypeCodec.computeLoadFunction('load_val_b', 'load_raw_b', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_c', 'store_raw_c', dtype)}

fn get_bshape(d: u32) -> u32 {
  return metadata.batch_shape[d / 4u][d % 4u];
}
fn get_bstride_a(d: u32) -> i32 {
  return metadata.batch_strides_a[d / 4u][d % 4u];
}
fn get_bstride_b(d: u32) -> i32 {
  return metadata.batch_strides_b[d / 4u][d % 4u];
}
fn get_bstride_out(d: u32) -> i32 {
  return metadata.batch_strides_out[d / 4u][d % 4u];
}

var<workgroup> tile_a: array<array<$valType, 16>, 16>;
var<workgroup> tile_b: array<array<$valType, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let batch_idx = global_id.z + metadata.batch_offset;
  if (batch_idx >= metadata.batch_count) {
    return;
  }
  let row = global_id.y;
  let col = global_id.x;
  let l_row = local_id.y;
  let l_col = local_id.x;

  var rem = batch_idx;
  var boff_a: i32 = i32(metadata.offset_a);
  var boff_b: i32 = i32(metadata.offset_b);
  var boff_out: i32 = i32(metadata.offset_out);
  for (var i: u32 = 0u; i < metadata.batch_rank; i = i + 1u) {
    let d = metadata.batch_rank - 1u - i;
    let dim_sz = get_bshape(d);
    let coord = i32(rem % dim_sz);
    rem = rem / dim_sz;
    boff_a = boff_a + coord * get_bstride_a(d);
    boff_b = boff_b + coord * get_bstride_b(d);
    boff_out = boff_out + coord * get_bstride_out(d);
  }

  var acc: $valType = $zeroExpr;
  let num_tiles = (metadata.K + 15u) / 16u;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let a_col = t * 16u + l_col;
    if (row < metadata.M && a_col < metadata.K) {
      let a_idx = u32(boff_a + i32(row) * metadata.stride_a_m + i32(a_col) * metadata.stride_a_k);
      tile_a[l_row][l_col] = load_val_a(a_idx);
    } else {
      tile_a[l_row][l_col] = $zeroExpr;
    }

    let b_row = t * 16u + l_row;
    if (b_row < metadata.K && col < metadata.N) {
      let b_idx = u32(boff_b + i32(b_row) * metadata.stride_b_k + i32(col) * metadata.stride_b_n);
      tile_b[l_row][l_col] = load_val_b(b_idx);
    } else {
      tile_b[l_row][l_col] = $zeroExpr;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let a = tile_a[l_row][k];
      let b = tile_b[k][l_col];
      let prod_val = $mulExpr;
      acc = $addExpr;
    }

    workgroupBarrier();
  }

  if (row < metadata.M && col < metadata.N) {
    let out_idx = u32(boff_out + i32(row) * metadata.stride_out_m + i32(col) * metadata.stride_out_n);
    store_val_c(out_idx, acc);
  }
}
''';

    return _matmulCache[dtype] = WgslShaderModule(
      name: 'universal_matmul_${dtype.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.tiled2D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'mat_a',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'mat_b',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'mat_c',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'MatmulMetadata',
        ),
      ],
    );
  }

  /// Generates a 1D fill compute shader for [dtype].
  static WgslShaderModule fillShader(DType dtype) {
    if (_fillCache[dtype] case final cached?) {
      return cached;
    }
    final valExpr = WgslDTypeCodec.rawFromUniformWords(
      dtype,
      'uniforms.val0',
      'uniforms.val1',
      'uniforms.val2',
      'uniforms.val3',
    );
    final code =
        '''
struct FillUniforms {
  count: u32,
  rank: u32,
  offset: u32,
  pad0: u32,
  val0: u32,
  val1: u32,
  val2: u32,
  val3: u32,
  shape: array<vec4<u32>, 2>,
  strides: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.writeBindingDecl(0, 'dst', dtype)}
@group(0) @binding(1) var<uniform> uniforms: FillUniforms;

${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= uniforms.count) {
    return;
  }
  if (uniforms.pad0 == 1u) {
    store_raw_dst(uniforms.offset + idx, $valExpr);
    return;
  }
  var rem = idx;
  var off_out: i32 = i32(uniforms.offset);
  for (var i: u32 = 0u; i < uniforms.rank; i = i + 1u) {
    let d = uniforms.rank - 1u - i;
    let dim_sz = uniforms.shape[d / 4u][d % 4u];
    let coord = i32(rem % dim_sz);
    rem = rem / dim_sz;
    off_out = off_out + coord * uniforms.strides[d / 4u][d % 4u];
  }
  store_raw_dst(u32(off_out), $valExpr);
}
''';

    return _fillCache[dtype] = WgslShaderModule(
      name: 'fill_${dtype.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'dst',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 1,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'FillUniforms',
        ),
      ],
    );
  }

  /// Generates a sequence or matrix generation compute shader (`arange`,
  /// `linspace`, `logspace`, `geomspace`, `eye`, `tri`) for [dtype].
  static WgslShaderModule generatorShader(DType dtype) {
    if (_generatorCache[dtype] case final cached?) {
      return cached;
    }
    final convertExpr = switch (dtype) {
      DType.complex64 || DType.complex128 => 'vec2<f32>(val_f32, 0.0)',
      DType.boolean => 'select(0u, 1u, val_f32 != 0.0)',
      DType.int64 => 'f32_to_i64(round(val_f32))',
      DType.uint64 => 'f32_to_u64(max(0.0, round(val_f32)))',
      DType.int32 || DType.int16 || DType.int8 => 'i32(round(val_f32))',
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 => 'u32(max(0.0, round(val_f32)))',
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => 'val_f32',
    };

    final code =
        '''
${WgslDTypeCodec.wgslNumericHelpers}

struct GeneratorUniforms {
  count: u32,
  mode: u32,
  rows: u32,
  cols: u32,
  k_offset: i32,
  start_val: f32,
  step_val: f32,
  base_val: f32,
};

${WgslDTypeCodec.writeBindingDecl(0, 'dst', dtype)}
@group(0) @binding(1) var<uniform> uniforms: GeneratorUniforms;

${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= uniforms.count) {
    return;
  }
  var val_f32: f32 = 0.0;
  if (uniforms.mode == 0u) {
    // arange / linspace: start + idx * step
    val_f32 = uniforms.start_val + f32(idx) * uniforms.step_val;
  } else if (uniforms.mode == 1u) {
    // logspace: pow(base, start + idx * step)
    let exp_v = uniforms.start_val + f32(idx) * uniforms.step_val;
    val_f32 = pow(uniforms.base_val, exp_v);
  } else if (uniforms.mode == 2u) {
    // eye: 1 on diagonal k_offset, else 0
    let r = i32(idx / uniforms.cols);
    let c = i32(idx % uniforms.cols);
    val_f32 = select(0.0, 1.0, (c - r) == uniforms.k_offset);
  } else if (uniforms.mode == 3u) {
    // tri: 1 where c <= r + k_offset, else 0
    let r = i32(idx / uniforms.cols);
    let c = i32(idx % uniforms.cols);
    val_f32 = select(0.0, 1.0, c <= (r + uniforms.k_offset));
  }
  store_val_dst(idx, $convertExpr);
}
''';

    return _generatorCache[dtype] = WgslShaderModule(
      name: 'generator_${dtype.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'dst',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 1,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'GeneratorUniforms',
        ),
      ],
    );
  }
}
