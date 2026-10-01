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

/// Universal 15-[DType] WGSL compute shader generators for indexing, selection,
/// stream compaction, and tensor manipulation kernels.
extension type const WgslIndexingKernels._(Object? _) {
  static final Map<DType, WgslShaderModule> _whereCache = {};
  static final Map<DType, WgslShaderModule> _tileCache = {};
  static final Map<(DType, DType), WgslShaderModule> _takeCache = {};
  static final Map<(DType, DType), WgslShaderModule> _putCache = {};
  static final Map<(DType, DType, bool), WgslShaderModule> _alongAxisCache = {};
  static final Map<DType, WgslShaderModule> _padCache = {};

  static String _indexLoadFunction(
    String functionName,
    String rawLoadName,
    DType indexDType,
  ) {
    final body = switch (indexDType) {
      DType.int8 => 'let r = $rawLoadName(idx); return (i32(r << 24u) >> 24u);',
      DType.int16 =>
        'let r = $rawLoadName(idx); return (i32(r << 16u) >> 16u);',
      DType.int32 => 'return bitcast<i32>($rawLoadName(idx));',
      DType.int64 ||
      DType.uint64 => 'return bitcast<i32>($rawLoadName(idx).x);',
      DType.uint8 ||
      DType.uint16 ||
      DType.uint32 ||
      DType.boolean => 'return i32($rawLoadName(idx));',
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 ||
      DType.complex64 ||
      DType.complex128 => 'return i32($rawLoadName(idx));',
    };
    return '''
fn $functionName(idx: u32) -> i32 {
  $body
}
''';
  }

  /// Generates a universal strided `where(cond, x, y)` shader for [dtype].
  static WgslShaderModule whereShader(DType dtype) {
    if (_whereCache[dtype] case final cached?) {
      return cached;
    }
    final wgslDType = WgslDType.fromDType(dtype);
    final code =
        '''
struct WhereMetadata {
  total_elements: u32,
  rank: u32,
  offset_cond: u32,
  offset_x: u32,
  offset_y: u32,
  offset_out: u32,
  pad0: u32,
  pad1: u32,
  shape: array<vec4<u32>, 2>,
  strides_cond: array<vec4<i32>, 2>,
  strides_x: array<vec4<i32>, 2>,
  strides_y: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

@group(0) @binding(0) var<storage, read> cond: array<u32>;
${WgslDTypeCodec.readBindingDecl(1, 'x', dtype)}
${WgslDTypeCodec.readBindingDecl(2, 'y', dtype)}
${WgslDTypeCodec.writeBindingDecl(3, 'dst', dtype)}
@group(0) @binding(4) var<uniform> metadata: WhereMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_x', 'x', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_y', 'y', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

fn get_shape_dim(params: WhereMetadata, d: u32) -> u32 {
  return params.shape[d / 4u][d % 4u];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem = idx;
  var off_cond: i32 = i32(metadata.offset_cond);
  var off_x: i32 = i32(metadata.offset_x);
  var off_y: i32 = i32(metadata.offset_y);
  var off_out: i32 = i32(metadata.offset_out);

  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_size = get_shape_dim(metadata, d);
    let coord = i32(rem % dim_size);
    rem = rem / dim_size;
    off_cond = off_cond + coord * metadata.strides_cond[d / 4u][d % 4u];
    off_x = off_x + coord * metadata.strides_x[d / 4u][d % 4u];
    off_y = off_y + coord * metadata.strides_y[d / 4u][d % 4u];
    off_out = off_out + coord * metadata.strides_out[d / 4u][d % 4u];
  }

  let cond_idx = u32(off_cond);
  let c_byte = (cond[cond_idx >> 2u] >> ((cond_idx & 3u) * 8u)) & 0xFFu;
  let chosen = select(load_raw_y(u32(off_y)), load_raw_x(u32(off_x)), c_byte != 0u);
  store_raw_dst(u32(off_out), chosen);
}
''';

    return _whereCache[dtype] = WgslShaderModule(
      name: 'where_${wgslDType.wgslType}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        const WgslBinding(
          group: 0,
          binding: 0,
          name: 'cond',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'x',
          dtype: wgslDType,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'y',
          dtype: wgslDType,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 3,
          name: 'dst',
          dtype: wgslDType,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 4,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'WhereMetadata',
        ),
      ],
    );
  }

  /// Generates a universal strided `tile` shader for [dtype].
  static WgslShaderModule tileShader(DType dtype) {
    if (_tileCache[dtype] case final cached?) {
      return cached;
    }
    final wgslDType = WgslDType.fromDType(dtype);
    final code =
        '''
struct TileMetadata {
  total_elements: u32,
  rank: u32,
  offset_a: u32,
  offset_out: u32,
  shape_in: array<vec4<u32>, 2>,
  shape_out: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: TileMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

fn get_shape_dim(params: TileMetadata, d: u32) -> u32 {
  return params.shape_out[d / 4u][d % 4u];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem = idx;
  var off_a: i32 = i32(metadata.offset_a);
  var off_out: i32 = i32(metadata.offset_out);

  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let out_dim = get_shape_dim(metadata, d);
    let in_dim = metadata.shape_in[d / 4u][d % 4u];
    let coord_out = i32(rem % out_dim);
    rem = rem / out_dim;
    let coord_in = coord_out % i32(in_dim);
    off_a = off_a + coord_in * metadata.strides_a[d / 4u][d % 4u];
    off_out = off_out + coord_out * metadata.strides_out[d / 4u][d % 4u];
  }

  store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
}
''';

    return _tileCache[dtype] = WgslShaderModule(
      name: 'tile_${wgslDType.wgslType}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: wgslDType,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: wgslDType,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'TileMetadata',
        ),
      ],
    );
  }

  /// Generates a universal `take` compute shader with out-of-bounds index
  /// detection for [dtype] and [indexDType].
  static WgslShaderModule takeShader(DType dtype, DType indexDType) {
    final key = (dtype, indexDType);
    if (_takeCache[key] case final cached?) {
      return cached;
    }
    final code =
        '''
struct TakeMetadata {
  total_elements: u32,
  rank_a: u32,
  rank_idx: u32,
  rank_out: u32,
  axis: u32,
  axis_size: u32,
  idx_size: u32,
  inner_size: u32,
  offset_a: u32,
  offset_idx: u32,
  offset_out: u32,
  pad0: u32,
  shape_a: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  shape_idx: array<vec4<u32>, 2>,
  strides_idx: array<vec4<i32>, 2>,
  shape_out: array<vec4<u32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.readBindingDecl(1, 'indices', indexDType)}
${WgslDTypeCodec.writeBindingDecl(2, 'dst', dtype)}
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> metadata: TakeMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_idx', 'indices', indexDType)}
${_indexLoadFunction('load_idx', 'load_raw_idx', indexDType)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem_out = idx;
  var off_out: i32 = i32(metadata.offset_out);
  for (var i: u32 = 0u; i < metadata.rank_out; i = i + 1u) {
    let d = metadata.rank_out - 1u - i;
    let dim_sz = metadata.shape_out[d / 4u][d % 4u];
    let coord = i32(rem_out % dim_sz);
    rem_out = rem_out / dim_sz;
    off_out = off_out + coord * metadata.strides_out[d / 4u][d % 4u];
  }

  let limit = i32(metadata.axis_size);
  if (metadata.axis == 99u) {
    var rem_idx = idx;
    var off_idx: i32 = i32(metadata.offset_idx);
    for (var i: u32 = 0u; i < metadata.rank_idx; i = i + 1u) {
      let d = metadata.rank_idx - 1u - i;
      let dim_sz = metadata.shape_idx[d / 4u][d % 4u];
      let coord = i32(rem_idx % dim_sz);
      rem_idx = rem_idx / dim_sz;
      off_idx = off_idx + coord * metadata.strides_idx[d / 4u][d % 4u];
    }
    var raw_i = load_idx(u32(off_idx));
    if (raw_i < -limit || raw_i >= limit) {
      atomicStore(&status[0], 1u);
      atomicStore(&status[1], bitcast<u32>(raw_i));
      return;
    }
    if (raw_i < 0) {
      raw_i = raw_i + limit;
    }
    var rem_a = u32(raw_i);
    var off_a: i32 = i32(metadata.offset_a);
    for (var i: u32 = 0u; i < metadata.rank_a; i = i + 1u) {
      let d = metadata.rank_a - 1u - i;
      let dim_sz = metadata.shape_a[d / 4u][d % 4u];
      let coord = i32(rem_a % dim_sz);
      rem_a = rem_a / dim_sz;
      off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
    }
    store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
    return;
  }

  let inner_rem = idx % metadata.inner_size;
  let mid_outer = idx / metadata.inner_size;
  let idx_flat = mid_outer % metadata.idx_size;
  let outer_flat = mid_outer / metadata.idx_size;

  var rem_idx = idx_flat;
  var off_idx: i32 = i32(metadata.offset_idx);
  for (var i: u32 = 0u; i < metadata.rank_idx; i = i + 1u) {
    let d = metadata.rank_idx - 1u - i;
    let dim_sz = metadata.shape_idx[d / 4u][d % 4u];
    let coord = i32(rem_idx % dim_sz);
    rem_idx = rem_idx / dim_sz;
    off_idx = off_idx + coord * metadata.strides_idx[d / 4u][d % 4u];
  }

  var raw_i = load_idx(u32(off_idx));
  if (raw_i < -limit || raw_i >= limit) {
    atomicStore(&status[0], 1u);
    atomicStore(&status[1], bitcast<u32>(raw_i));
    return;
  }
  if (raw_i < 0) {
    raw_i = raw_i + limit;
  }

  var off_a: i32 = i32(metadata.offset_a) + raw_i * metadata.strides_a[metadata.axis / 4u][metadata.axis % 4u];
  var rem_in = inner_rem;
  for (var d_rev: u32 = 0u; d_rev < metadata.rank_a; d_rev = d_rev + 1u) {
    let d = metadata.rank_a - 1u - d_rev;
    if (d <= metadata.axis) {
      break;
    }
    let dim_sz = metadata.shape_a[d / 4u][d % 4u];
    let coord = i32(rem_in % dim_sz);
    rem_in = rem_in / dim_sz;
    off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
  }

  var rem_out_a = outer_flat;
  for (var d_rev: u32 = 0u; d_rev < metadata.axis; d_rev = d_rev + 1u) {
    let d = metadata.axis - 1u - d_rev;
    let dim_sz = metadata.shape_a[d / 4u][d % 4u];
    let coord = i32(rem_out_a % dim_sz);
    rem_out_a = rem_out_a / dim_sz;
    off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
  }

  store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
}
''';

    return _takeCache[key] = WgslShaderModule(
      name: 'take_${dtype.name}_${indexDType.name}',
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
          name: 'indices',
          dtype: WgslDType.fromDType(indexDType),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'status',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 4,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'TakeMetadata',
        ),
      ],
    );
  }

  /// Generates a universal flat `put` compute shader with out-of-bounds index
  /// detection for [dtype] and [indexDType].
  static WgslShaderModule putShader(DType dtype, DType indexDType) {
    final key = (dtype, indexDType);
    if (_putCache[key] case final cached?) {
      return cached;
    }
    final code =
        '''
struct PutMetadata {
  idx_count: u32,
  arr_size: u32,
  val_size: u32,
  rank_a: u32,
  rank_idx: u32,
  rank_val: u32,
  offset_a: u32,
  offset_idx: u32,
  offset_val: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  shape_a: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  shape_idx: array<vec4<u32>, 2>,
  strides_idx: array<vec4<i32>, 2>,
  shape_val: array<vec4<u32>, 2>,
  strides_val: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'indices', indexDType)}
${WgslDTypeCodec.readBindingDecl(1, 'values', dtype)}
${WgslDTypeCodec.writeBindingDecl(2, 'dst', dtype)}
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> metadata: PutMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_idx', 'indices', indexDType)}
${_indexLoadFunction('load_idx', 'load_raw_idx', indexDType)}
${WgslDTypeCodec.rawLoadFunction('load_raw_val', 'values', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.idx_count) {
    return;
  }

  var rem_idx = idx;
  var off_idx: i32 = i32(metadata.offset_idx);
  for (var i: u32 = 0u; i < metadata.rank_idx; i = i + 1u) {
    let d = metadata.rank_idx - 1u - i;
    let dim_sz = metadata.shape_idx[d / 4u][d % 4u];
    let coord = i32(rem_idx % dim_sz);
    rem_idx = rem_idx / dim_sz;
    off_idx = off_idx + coord * metadata.strides_idx[d / 4u][d % 4u];
  }

  let limit = i32(metadata.arr_size);
  var raw_i = load_idx(u32(off_idx));
  if (raw_i < -limit || raw_i >= limit) {
    atomicStore(&status[0], 1u);
    atomicStore(&status[1], bitcast<u32>(raw_i));
    return;
  }
  if (raw_i < 0) {
    raw_i = raw_i + limit;
  }

  var rem_a = u32(raw_i);
  var off_a: i32 = i32(metadata.offset_a);
  for (var i: u32 = 0u; i < metadata.rank_a; i = i + 1u) {
    let d = metadata.rank_a - 1u - i;
    let dim_sz = metadata.shape_a[d / 4u][d % 4u];
    let coord = i32(rem_a % dim_sz);
    rem_a = rem_a / dim_sz;
    off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
  }

  var rem_val = idx % metadata.val_size;
  var off_val: i32 = i32(metadata.offset_val);
  for (var i: u32 = 0u; i < metadata.rank_val; i = i + 1u) {
    let d = metadata.rank_val - 1u - i;
    let dim_sz = metadata.shape_val[d / 4u][d % 4u];
    let coord = i32(rem_val % dim_sz);
    rem_val = rem_val / dim_sz;
    off_val = off_val + coord * metadata.strides_val[d / 4u][d % 4u];
  }

  store_raw_dst(u32(off_a), load_raw_val(u32(off_val)));
}
''';

    return _putCache[key] = WgslShaderModule(
      name: 'put_${dtype.name}_${indexDType.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'indices',
          dtype: WgslDType.fromDType(indexDType),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'values',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'status',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 4,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'PutMetadata',
        ),
      ],
    );
  }

  /// Generates a universal `takeAlongAxis` or `putAlongAxis` compute shader
  /// with out-of-bounds index detection for [dtype] and [indexDType].
  static WgslShaderModule alongAxisShader({
    required DType dtype,
    required DType indexDType,
    required bool isPut,
  }) {
    final key = (dtype, indexDType, isPut);
    if (_alongAxisCache[key] case final cached?) {
      return cached;
    }
    final body = isPut
        ? 'store_raw_dst(u32(off_a), load_raw_src(u32(off_out)));'
        : 'store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));';
    final code =
        '''
struct AlongAxisMetadata {
  total_elements: u32,
  rank: u32,
  axis: u32,
  axis_dim_a: u32,
  offset_a: u32,
  offset_idx: u32,
  offset_out: u32,
  pad0: u32,
  shape_out: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_idx: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.readBindingDecl(1, 'indices', indexDType)}
${WgslDTypeCodec.writeBindingDecl(2, 'dst', dtype)}
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> metadata: AlongAxisMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_idx', 'indices', indexDType)}
${_indexLoadFunction('load_idx', 'load_raw_idx', indexDType)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem = idx;
  var off_idx: i32 = i32(metadata.offset_idx);
  var off_out: i32 = i32(metadata.offset_out);
  var base_off_a: i32 = i32(metadata.offset_a);

  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_size = metadata.shape_out[d / 4u][d % 4u];
    let coord = i32(rem % dim_size);
    rem = rem / dim_size;
    off_idx = off_idx + coord * metadata.strides_idx[d / 4u][d % 4u];
    off_out = off_out + coord * metadata.strides_out[d / 4u][d % 4u];
    if (d != metadata.axis) {
      base_off_a = base_off_a + coord * metadata.strides_a[d / 4u][d % 4u];
    }
  }

  let limit = i32(metadata.axis_dim_a);
  var target_coord = load_idx(u32(off_idx));
  if (target_coord < -limit || target_coord >= limit) {
    atomicStore(&status[0], 1u);
    atomicStore(&status[1], bitcast<u32>(target_coord));
    return;
  }
  if (target_coord < 0) {
    target_coord = target_coord + limit;
  }

  let off_a = base_off_a + target_coord * metadata.strides_a[metadata.axis / 4u][metadata.axis % 4u];
  $body
}
''';

    final prefix = isPut ? 'put_along_axis' : 'take_along_axis';
    return _alongAxisCache[key] = WgslShaderModule(
      name: '${prefix}_${dtype.name}_${indexDType.name}',
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
          name: 'indices',
          dtype: WgslDType.fromDType(indexDType),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'status',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 4,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'AlongAxisMetadata',
        ),
      ],
    );
  }

  /// Generates a universal `pad` compute shader supporting all 5 `PadMode`s
  /// and all 15 [DType]s.
  static WgslShaderModule padShader(DType dtype) {
    if (_padCache[dtype] case final cached?) {
      return cached;
    }
    final constValExpr = WgslDTypeCodec.rawFromUniformWords(
      dtype,
      'metadata.val0',
      'metadata.val1',
      'metadata.val2',
      'metadata.val3',
    );
    final code =
        '''
struct PadMetadata {
  total_elements: u32,
  rank: u32,
  offset_a: u32,
  offset_out: u32,
  mode: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  val0: u32,
  val1: u32,
  val2: u32,
  val3: u32,
  shape_in: array<vec4<u32>, 2>,
  shape_out: array<vec4<u32>, 2>,
  pad_before: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: PadMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

fn map_pad_coord(c: i32, pad_bef: i32, src_len: i32, mode: u32) -> i32 {
  var idx = c - pad_bef;
  if (mode == 1u) {
    return clamp(idx, 0, src_len - 1);
  }
  if (mode == 2u) {
    if (src_len <= 1) {
      return 0;
    }
    let period = 2 * (src_len - 1);
    idx = ((idx % period) + period) % period;
    if (idx >= src_len) {
      idx = period - idx;
    }
    return idx;
  }
  if (mode == 3u) {
    let period = 2 * src_len;
    idx = ((idx % period) + period) % period;
    if (idx >= src_len) {
      idx = period - 1 - idx;
    }
    return idx;
  }
  if (mode == 4u) {
    return ((idx % src_len) + src_len) % src_len;
  }
  return idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem = idx;
  var off_a: i32 = i32(metadata.offset_a);
  var off_out: i32 = i32(metadata.offset_out);
  var in_bounds: bool = true;

  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let out_dim = metadata.shape_out[d / 4u][d % 4u];
    let in_dim = i32(metadata.shape_in[d / 4u][d % 4u]);
    let p_bef = i32(metadata.pad_before[d / 4u][d % 4u]);
    let coord_out = i32(rem % out_dim);
    rem = rem / out_dim;

    off_out = off_out + coord_out * metadata.strides_out[d / 4u][d % 4u];
    let coord_in = map_pad_coord(coord_out, p_bef, in_dim, metadata.mode);
    if (coord_in < 0 || coord_in >= in_dim) {
      in_bounds = false;
    } else {
      off_a = off_a + coord_in * metadata.strides_a[d / 4u][d % 4u];
    }
  }

  if (in_bounds) {
    store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
  } else {
    store_raw_dst(u32(off_out), $constValExpr);
  }
}
''';

    return _padCache[dtype] = WgslShaderModule(
      name: 'pad_${dtype.name}',
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
          name: 'metadata',
          isUniform: true,
          customTypeName: 'PadMetadata',
        ),
      ],
    );
  }
}
