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
  static final Map<DType, WgslShaderModule> _axisSortCache = {};
  static final Map<(DType, DType), WgslShaderModule> _searchSortedCache = {};
  static final Map<DType, WgslShaderModule> _uniqueRowSortCache = {};
  static final Map<DType, WgslShaderModule> _uniqueMarkScanCache = {};
  static final Map<DType, WgslShaderModule> _uniqueScatterCache = {};
  static final Map<(DType, DType?), WgslShaderModule> _bincountCache = {};
  static final Map<(String, DType), WgslShaderModule> _cumulativeScanCache = {};
  static final Map<DType, WgslShaderModule> _diffCache = {};

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

  static String _rawComparisonHelpers(DType dtype) {
    final rawType = WgslDTypeCodec.rawStorageElementType(dtype);
    final ltBody = switch (dtype) {
      DType.float64 => 'return f64_total_lt(a, b);',
      DType.float32 => 'return f32_total_lt(bitcast<f32>(a), bitcast<f32>(b));',
      DType.float16 => 'return f32_total_lt(f16_to_f32(a), f16_to_f32(b));',
      DType.bfloat16 => 'return f32_total_lt(bf16_to_f32(a), bf16_to_f32(b));',
      DType.int64 => 'return i64_lt(a, b);',
      DType.uint64 => 'return u64_lt(a, b);',
      DType.int32 => 'return bitcast<i32>(a) < bitcast<i32>(b);',
      DType.int16 => 'return (i32(a << 16u) >> 16u) < (i32(b << 16u) >> 16u);',
      DType.int8 => 'return (i32(a << 24u) >> 24u) < (i32(b << 24u) >> 24u);',
      DType.uint32 || DType.uint16 || DType.uint8 => 'return a < b;',
      DType.boolean => 'return (a == 0u) && (b != 0u);',
      DType.complex64 =>
        '''
  let ar = bitcast<f32>(a.x);
  let ai = bitcast<f32>(a.y);
  let br = bitcast<f32>(b.x);
  let bi = bitcast<f32>(b.y);
  if (f32_total_lt(ar, br)) { return true; }
  if (f32_total_lt(br, ar)) { return false; }
  return f32_total_lt(ai, bi);''',
      DType.complex128 =>
        '''
  if (f64_total_lt(a.xy, b.xy)) { return true; }
  if (f64_total_lt(b.xy, a.xy)) { return false; }
  return f64_total_lt(a.zw, b.zw);''',
    };
    return '''
${WgslDTypeCodec.wgslNumericHelpers}

fn f64_total_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  let a_nan = ((a.y & 0x7FF00000u) == 0x7FF00000u) && (((a.y & 0x000FFFFFu) != 0u) || (a.x != 0u));
  let b_nan = ((b.y & 0x7FF00000u) == 0x7FF00000u) && (((b.y & 0x000FFFFFu) != 0u) || (b.x != 0u));
  if (a_nan || b_nan) {
    return !a_nan && b_nan;
  }
  let a_zero = ((a.y & 0x7FFFFFFFu) == 0u) && (a.x == 0u);
  let b_zero = ((b.y & 0x7FFFFFFFu) == 0u) && (b.x == 0u);
  if (a_zero && b_zero) {
    return false;
  }
  let a_neg = (a.y & 0x80000000u) != 0u;
  let b_neg = (b.y & 0x80000000u) != 0u;
  if (a_neg != b_neg) {
    return a_neg;
  }
  let eq_mag = (a.y == b.y) && (a.x == b.x);
  if (eq_mag) {
    return false;
  }
  let lt_mag = (a.y < b.y) || ((a.y == b.y) && (a.x < b.x));
  return select(lt_mag, !lt_mag, a_neg);
}

fn f32_total_lt(a: f32, b: f32) -> bool {
  let a_nan = f32_isnan(a);
  let b_nan = f32_isnan(b);
  if (a_nan || b_nan) {
    return !a_nan && b_nan;
  }
  return a < b;
}

fn raw_lt(a: $rawType, b: $rawType) -> bool {
  $ltBody
}

fn raw_eq(a: $rawType, b: $rawType) -> bool {
  return !raw_lt(a, b) && !raw_lt(b, a);
}
''';
  }

  /// Generates a workgroup-parallel bitonic + rank sort shader along an axis
  /// for [dtype], supporting `sort`, `argsort`, `partition`, `argpartition`,
  /// and `topk`.
  static WgslShaderModule axisSortShader(DType dtype) {
    if (_axisSortCache[dtype] case final cached?) {
      return cached;
    }
    final rawType = WgslDTypeCodec.rawStorageElementType(dtype);
    final rawZero = WgslDTypeCodec.rawZeroLiteral(dtype);
    final code =
        '''
struct SortMetadata {
  num_slices: u32,
  rank: u32,
  axis: u32,
  axis_size: u32,
  out_axis_size: u32,
  offset_src: u32,
  offset_dst_val: u32,
  offset_dst_idx: u32,
  stride_src_axis: i32,
  stride_dst_val_axis: i32,
  stride_dst_idx_axis: i32,
  descending: u32,
  write_values: u32,
  write_indices: u32,
  pad0: u32,
  pad1: u32,
  slice_shape: array<vec4<u32>, 2>,
  strides_src: array<vec4<i32>, 2>,
  strides_dst_val: array<vec4<i32>, 2>,
  strides_dst_idx: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst_vals', dtype)}
${WgslDTypeCodec.writeBindingDecl(2, 'dst_idx', DType.int64)}
@group(0) @binding(3) var<uniform> metadata: SortMetadata;

${_rawComparisonHelpers(dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_val', 'dst_vals', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_idx', 'dst_idx', DType.int64)}

var<workgroup> s_vals: array<$rawType, 512>;
var<workgroup> s_idx: array<u32, 512>;

fn pair_lt(va: $rawType, ia: u32, vb: $rawType, ib: u32, desc: bool) -> bool {
  if (ia == 0xFFFFFFFFu && ib == 0xFFFFFFFFu) {
    return false;
  }
  if (ia == 0xFFFFFFFFu) {
    return false;
  }
  if (ib == 0xFFFFFFFFu) {
    return true;
  }
  if (desc) {
    if (raw_lt(vb, va)) {
      return true;
    }
    if (raw_lt(va, vb)) {
      return false;
    }
    return ia < ib;
  } else {
    if (raw_lt(va, vb)) {
      return true;
    }
    if (raw_lt(vb, va)) {
      return false;
    }
    return ia < ib;
  }
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>
) {
  let slice_idx = wg_id.y * num_wg.x + wg_id.x;
  let tid = local_id.x;
  let is_active = slice_idx < metadata.num_slices;

  var base_src: i32 = i32(metadata.offset_src);
  var base_dst_val: i32 = i32(metadata.offset_dst_val);
  var base_dst_idx: i32 = i32(metadata.offset_dst_idx);

  if (is_active) {
    var rem = slice_idx;
    for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
      let d = metadata.rank - 1u - i;
      let dim_sz = metadata.slice_shape[d / 4u][d % 4u];
      let coord = i32(rem % dim_sz);
      rem = rem / dim_sz;
      base_src = base_src + coord * metadata.strides_src[d / 4u][d % 4u];
      base_dst_val = base_dst_val + coord * metadata.strides_dst_val[d / 4u][d % 4u];
      base_dst_idx = base_dst_idx + coord * metadata.strides_dst_idx[d / 4u][d % 4u];
    }
  }

  let n = metadata.axis_size;
  let desc = metadata.descending != 0u;

  if (n <= 512u) {
    var pow2: u32 = 1u;
    loop {
      if (pow2 >= n) {
        break;
      }
      pow2 = pow2 << 1u;
    }

    if (is_active && tid < n) {
      let off0 = base_src + i32(tid) * metadata.stride_src_axis;
      s_vals[tid] = load_raw_src(u32(off0));
      s_idx[tid] = tid;
    } else {
      s_vals[tid] = $rawZero;
      s_idx[tid] = 0xFFFFFFFFu;
    }
    let tid2 = tid + 256u;
    if (is_active && tid2 < n) {
      let off1 = base_src + i32(tid2) * metadata.stride_src_axis;
      s_vals[tid2] = load_raw_src(u32(off1));
      s_idx[tid2] = tid2;
    } else {
      s_vals[tid2] = $rawZero;
      s_idx[tid2] = 0xFFFFFFFFu;
    }
    workgroupBarrier();

    var k_step: u32 = 2u;
    loop {
      if (k_step > pow2) {
        break;
      }
      var j_step: u32 = k_step >> 1u;
      loop {
        if (j_step == 0u) {
          break;
        }
        let low_mask = j_step - 1u;
        let i0 = ((tid & ~low_mask) << 1u) | (tid & low_mask);
        let i1 = i0 ^ j_step;
        if (i1 < pow2) {
          let dir_up = (i0 & k_step) == 0u;
          let va = s_vals[i0];
          let ia = s_idx[i0];
          let vb = s_vals[i1];
          let ib = s_idx[i1];
          let a_before_b = pair_lt(va, ia, vb, ib, desc);
          if (a_before_b != dir_up) {
            s_vals[i0] = vb;
            s_idx[i0] = ib;
            s_vals[i1] = va;
            s_idx[i1] = ia;
          }
        }
        workgroupBarrier();
        j_step = j_step >> 1u;
      }
      k_step = k_step << 1u;
    }

    if (is_active) {
      let out_n = metadata.out_axis_size;
      if (tid < out_n) {
        if (metadata.write_values != 0u) {
          let dst_v_off = base_dst_val + i32(tid) * metadata.stride_dst_val_axis;
          store_raw_val(u32(dst_v_off), s_vals[tid]);
        }
        if (metadata.write_indices != 0u) {
          let dst_i_off = base_dst_idx + i32(tid) * metadata.stride_dst_idx_axis;
          store_raw_idx(u32(dst_i_off), vec2<u32>(s_idx[tid], 0u));
        }
      }
      if (tid2 < out_n) {
        if (metadata.write_values != 0u) {
          let dst_v_off = base_dst_val + i32(tid2) * metadata.stride_dst_val_axis;
          store_raw_val(u32(dst_v_off), s_vals[tid2]);
        }
        if (metadata.write_indices != 0u) {
          let dst_i_off = base_dst_idx + i32(tid2) * metadata.stride_dst_idx_axis;
          store_raw_idx(u32(dst_i_off), vec2<u32>(s_idx[tid2], 0u));
        }
      }
    }
  } else if (is_active) {
    let out_n = metadata.out_axis_size;
    var i = tid;
    loop {
      if (i >= n) {
        break;
      }
      let off_i = base_src + i32(i) * metadata.stride_src_axis;
      let val_i = load_raw_src(u32(off_i));
      var rnk: u32 = 0u;
      for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (j != i) {
          let off_j = base_src + i32(j) * metadata.stride_src_axis;
          let val_j = load_raw_src(u32(off_j));
          if (pair_lt(val_j, j, val_i, i, desc)) {
            rnk = rnk + 1u;
          }
        }
      }
      if (rnk < out_n) {
        if (metadata.write_values != 0u) {
          let dst_v_off = base_dst_val + i32(rnk) * metadata.stride_dst_val_axis;
          store_raw_val(u32(dst_v_off), val_i);
        }
        if (metadata.write_indices != 0u) {
          let dst_i_off = base_dst_idx + i32(rnk) * metadata.stride_dst_idx_axis;
          store_raw_idx(u32(dst_i_off), vec2<u32>(i, 0u));
        }
      }
      i = i + 256u;
    }
  }
}
''';

    return _axisSortCache[dtype] = WgslShaderModule(
      name: 'axis_sort_${dtype.name}',
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
          name: 'dst_vals',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst_idx',
          dtype: WgslDType.fromDType(DType.int64),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'SortMetadata',
        ),
      ],
    );
  }

  /// Generates a parallel binary-search `searchsorted` shader for [dtype] and
  /// [sorterDType], writing 64-bit integer insertion indices.
  static WgslShaderModule searchSortedShader({
    required DType dtype,
    required DType sorterDType,
  }) {
    final key = (dtype, sorterDType);
    if (_searchSortedCache[key] case final cached?) {
      return cached;
    }
    final code =
        '''
struct SearchSortedMetadata {
  total_v: u32,
  rank_v: u32,
  len_a: u32,
  side: u32,
  has_sorter: u32,
  offset_a: u32,
  offset_v: u32,
  offset_sorter: u32,
  offset_out: u32,
  stride_a: i32,
  stride_sorter: i32,
  pad0: u32,
  shape_v: array<vec4<u32>, 2>,
  strides_v: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'arr', dtype)}
${WgslDTypeCodec.readBindingDecl(1, 'vals', dtype)}
${WgslDTypeCodec.readBindingDecl(2, 'sorter', sorterDType)}
${WgslDTypeCodec.writeBindingDecl(3, 'dst', DType.int64)}
@group(0) @binding(4) var<uniform> metadata: SearchSortedMetadata;

${_rawComparisonHelpers(dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_a', 'arr', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_v', 'vals', dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_sorter', 'sorter', sorterDType)}
${_indexLoadFunction('load_sorter_idx', 'load_raw_sorter', sorterDType)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', DType.int64)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_v) {
    return;
  }

  var rem = idx;
  var off_v: i32 = i32(metadata.offset_v);
  var off_out: i32 = i32(metadata.offset_out);

  for (var i: u32 = 0u; i < metadata.rank_v; i = i + 1u) {
    let d = metadata.rank_v - 1u - i;
    let dim_sz = metadata.shape_v[d / 4u][d % 4u];
    let coord = i32(rem % dim_sz);
    rem = rem / dim_sz;
    off_v = off_v + coord * metadata.strides_v[d / 4u][d % 4u];
    off_out = off_out + coord * metadata.strides_out[d / 4u][d % 4u];
  }

  let val_v = load_raw_v(u32(off_v));
  var lo: u32 = 0u;
  var hi: u32 = metadata.len_a;
  loop {
    if (lo >= hi) {
      break;
    }
    let mid = lo + ((hi - lo) >> 1u);
    var a_coord = i32(mid);
    if (metadata.has_sorter != 0u) {
      let s_off = i32(metadata.offset_sorter) + i32(mid) * metadata.stride_sorter;
      a_coord = load_sorter_idx(u32(s_off));
      if (a_coord < 0) {
        a_coord = a_coord + i32(metadata.len_a);
      }
    }
    let a_off = i32(metadata.offset_a) + a_coord * metadata.stride_a;
    let val_a = load_raw_a(u32(a_off));
    let go_right = select(
      !raw_lt(val_v, val_a),
      raw_lt(val_a, val_v),
      metadata.side == 0u
    );
    if (go_right) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  store_raw_dst(u32(off_out), vec2<u32>(lo, 0u));
}
''';

    return _searchSortedCache[key] = WgslShaderModule(
      name: 'searchsorted_${dtype.name}_${sorterDType.name}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'arr',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'vals',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'sorter',
          dtype: WgslDType.fromDType(sorterDType),
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 3,
          name: 'dst',
          dtype: WgslDType.fromDType(DType.int64),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 4,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'SearchSortedMetadata',
        ),
      ],
    );
  }

  /// Generates a row-lexicographical sort shader for `unique` (supporting both
  /// 1-D elements when `row_len == 1` and N-D slices when `row_len > 1`).
  static WgslShaderModule uniqueRowSortShader(DType dtype) {
    if (_uniqueRowSortCache[dtype] case final cached?) {
      return cached;
    }
    final code =
        '''
struct UniqueSortMetadata {
  num_rows: u32,
  row_len: u32,
  pad0: u32,
  pad1: u32,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
@group(0) @binding(1) var<storage, read_write> sorted_idx: array<u32>;
@group(0) @binding(2) var<uniform> metadata: UniqueSortMetadata;

${_rawComparisonHelpers(dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}

var<workgroup> s_idx: array<u32, 512>;

fn row_lt(ra: u32, rb: u32, row_len: u32) -> bool {
  let base_a = ra * row_len;
  let base_b = rb * row_len;
  for (var k: u32 = 0u; k < row_len; k = k + 1u) {
    let va = load_raw_src(base_a + k);
    let vb = load_raw_src(base_b + k);
    if (raw_lt(va, vb)) {
      return true;
    }
    if (raw_lt(vb, va)) {
      return false;
    }
  }
  return false;
}

fn row_pair_lt(ia: u32, ib: u32, row_len: u32) -> bool {
  if (ia == 0xFFFFFFFFu && ib == 0xFFFFFFFFu) {
    return false;
  }
  if (ia == 0xFFFFFFFFu) {
    return false;
  }
  if (ib == 0xFFFFFFFFu) {
    return true;
  }
  if (row_lt(ia, ib, row_len)) {
    return true;
  }
  if (row_lt(ib, ia, row_len)) {
    return false;
  }
  return ia < ib;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
  let tid = local_id.x;
  let n = metadata.num_rows;
  let row_len = metadata.row_len;

  if (n <= 512u) {
    var pow2: u32 = 1u;
    loop {
      if (pow2 >= n) {
        break;
      }
      pow2 = pow2 << 1u;
    }
    s_idx[tid] = select(0xFFFFFFFFu, tid, tid < n);
    let tid2 = tid + 256u;
    s_idx[tid2] = select(0xFFFFFFFFu, tid2, tid2 < n);
    workgroupBarrier();

    var k_step: u32 = 2u;
    loop {
      if (k_step > pow2) {
        break;
      }
      var j_step: u32 = k_step >> 1u;
      loop {
        if (j_step == 0u) {
          break;
        }
        let low_mask = j_step - 1u;
        let i0 = ((tid & ~low_mask) << 1u) | (tid & low_mask);
        let i1 = i0 ^ j_step;
        if (i1 < pow2) {
          let dir_up = (i0 & k_step) == 0u;
          let ia = s_idx[i0];
          let ib = s_idx[i1];
          let a_before_b = row_pair_lt(ia, ib, row_len);
          if (a_before_b != dir_up) {
            s_idx[i0] = ib;
            s_idx[i1] = ia;
          }
        }
        workgroupBarrier();
        j_step = j_step >> 1u;
      }
      k_step = k_step << 1u;
    }

    if (tid < n) {
      sorted_idx[tid] = s_idx[tid];
    }
    if (tid2 < n) {
      sorted_idx[tid2] = s_idx[tid2];
    }
  } else {
    var i = tid;
    loop {
      if (i >= n) {
        break;
      }
      var rnk: u32 = 0u;
      for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (j != i && row_pair_lt(j, i, row_len)) {
          rnk = rnk + 1u;
        }
      }
      sorted_idx[rnk] = i;
      i = i + 256u;
    }
  }
}
''';

    return _uniqueRowSortCache[dtype] = WgslShaderModule(
      name: 'unique_row_sort_${dtype.name}',
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
        const WgslBinding(
          group: 0,
          binding: 1,
          name: 'sorted_idx',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'UniqueSortMetadata',
        ),
      ],
    );
  }

  /// Generates a parallel prefix-scan shader marking unique group boundaries,
  /// group IDs, head positions, and total unique count for [dtype].
  static WgslShaderModule uniqueMarkScanShader(DType dtype) {
    if (_uniqueMarkScanCache[dtype] case final cached?) {
      return cached;
    }
    final code =
        '''
struct UniqueScanMetadata {
  num_rows: u32,
  row_len: u32,
  pad0: u32,
  pad1: u32,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
@group(0) @binding(1) var<storage, read> sorted_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> group_id: array<u32>;
@group(0) @binding(3) var<storage, read_write> head_pos: array<u32>;
@group(0) @binding(4) var<storage, read_write> count_out: array<u32>;
@group(0) @binding(5) var<uniform> metadata: UniqueScanMetadata;

${_rawComparisonHelpers(dtype)}
${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}

var<workgroup> s_scan: array<u32, 256>;
var<workgroup> s_carry: u32;

fn row_eq(ra: u32, rb: u32, row_len: u32) -> bool {
  let base_a = ra * row_len;
  let base_b = rb * row_len;
  for (var k: u32 = 0u; k < row_len; k = k + 1u) {
    let va = load_raw_src(base_a + k);
    let vb = load_raw_src(base_b + k);
    if (!raw_eq(va, vb)) {
      return false;
    }
  }
  return true;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
  let tid = local_id.x;
  let n = metadata.num_rows;
  let row_len = metadata.row_len;

  if (tid == 0u) {
    s_carry = 0u;
    if (n == 0u) {
      count_out[0] = 0u;
    }
  }
  workgroupBarrier();

  let num_blocks = (n + 255u) / 256u;
  for (var blk: u32 = 0u; blk < num_blocks; blk = blk + 1u) {
    let i = blk * 256u + tid;
    var is_head: u32 = 0u;
    if (i < n) {
      if (i == 0u) {
        is_head = 1u;
      } else {
        let cur_r = sorted_idx[i];
        let prev_r = sorted_idx[i - 1u];
        is_head = select(1u, 0u, row_eq(cur_r, prev_r, row_len));
      }
    }
    var val = is_head;
    if (tid == 0u && blk > 0u) {
      val = val + s_carry;
    }
    s_scan[tid] = val;
    workgroupBarrier();

    var step_off: u32 = 1u;
    loop {
      if (step_off >= 256u) {
        break;
      }
      var merged = s_scan[tid];
      if (tid >= step_off) {
        merged = s_scan[tid - step_off] + merged;
      }
      workgroupBarrier();
      s_scan[tid] = merged;
      workgroupBarrier();
      step_off = step_off << 1u;
    }

    let prefix = s_scan[tid];
    if (i < n) {
      let gid = prefix - 1u;
      group_id[i] = gid;
      if (is_head != 0u) {
        head_pos[gid] = i;
      }
      if (i + 1u == n) {
        count_out[0] = prefix;
      }
    }
    if (tid == 255u) {
      s_carry = prefix;
    }
    workgroupBarrier();
  }
}
''';

    return _uniqueMarkScanCache[dtype] = WgslShaderModule(
      name: 'unique_mark_scan_${dtype.name}',
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
        const WgslBinding(
          group: 0,
          binding: 1,
          name: 'sorted_idx',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.read,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'group_id',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'head_pos',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 4,
          name: 'count_out',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 5,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'UniqueScanMetadata',
        ),
      ],
    );
  }

  /// Generates a scatter shader for `uniqueAll` populating unique `values`,
  /// first-occurrence `indices` (`Int64`), `inverse` (`Int64`), and `counts`
  /// (`Int64`).
  static WgslShaderModule uniqueScatterShader(DType dtype) {
    if (_uniqueScatterCache[dtype] case final cached?) {
      return cached;
    }
    final code =
        '''
struct UniqueScatterMetadata {
  num_rows: u32,
  num_unique: u32,
  row_len: u32,
  pad0: u32,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
@group(0) @binding(1) var<storage, read> sorted_idx: array<u32>;
@group(0) @binding(2) var<storage, read> group_id: array<u32>;
@group(0) @binding(3) var<storage, read> head_pos: array<u32>;
${WgslDTypeCodec.writeBindingDecl(4, 'out_values', dtype)}
${WgslDTypeCodec.writeBindingDecl(5, 'out_indices', DType.int64)}
${WgslDTypeCodec.writeBindingDecl(6, 'out_inverse', DType.int64)}
${WgslDTypeCodec.writeBindingDecl(7, 'out_counts', DType.int64)}
@group(0) @binding(8) var<uniform> metadata: UniqueScatterMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_values', 'out_values', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_indices', 'out_indices', DType.int64)}
${WgslDTypeCodec.rawStoreFunction('store_raw_inverse', 'out_inverse', DType.int64)}
${WgslDTypeCodec.rawStoreFunction('store_raw_counts', 'out_counts', DType.int64)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;

  if (idx < metadata.num_rows) {
    let orig_row = sorted_idx[idx];
    let gid = group_id[idx];
    store_raw_inverse(orig_row, vec2<u32>(gid, 0u));
  }

  if (idx < metadata.num_unique) {
    let start_i = head_pos[idx];
    var end_i = metadata.num_rows;
    if (idx + 1u < metadata.num_unique) {
      end_i = head_pos[idx + 1u];
    }
    let first_orig = sorted_idx[start_i];
    store_raw_indices(idx, vec2<u32>(first_orig, 0u));
    store_raw_counts(idx, vec2<u32>(end_i - start_i, 0u));
  }

  let total_val_elems = metadata.num_unique * metadata.row_len;
  if (idx < total_val_elems) {
    let g = idx / metadata.row_len;
    let col = idx % metadata.row_len;
    let orig_row = sorted_idx[head_pos[g]];
    let val = load_raw_src(orig_row * metadata.row_len + col);
    store_raw_values(idx, val);
  }
}
''';

    return _uniqueScatterCache[dtype] = WgslShaderModule(
      name: 'unique_scatter_${dtype.name}',
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
        const WgslBinding(
          group: 0,
          binding: 1,
          name: 'sorted_idx',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.read,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'group_id',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.read,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'head_pos',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 4,
          name: 'out_values',
          dtype: WgslDType.fromDType(dtype),
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 5,
          name: 'out_indices',
          dtype: WgslDType.fromDType(DType.int64),
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 6,
          name: 'out_inverse',
          dtype: WgslDType.fromDType(DType.int64),
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 7,
          name: 'out_counts',
          dtype: WgslDType.fromDType(DType.int64),
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 8,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'UniqueScatterMetadata',
        ),
      ],
    );
  }

  /// Generates a `bincount` shader for 1-D integer input [xDType] and optional
  /// [weightsDType], producing `Int64` counts or `Float64` weighted sums.
  static WgslShaderModule bincountShader({
    required DType xDType,
    DType? weightsDType,
  }) {
    final key = (xDType, weightsDType);
    if (_bincountCache[key] case final cached?) {
      return cached;
    }
    final hasWeights = weightsDType != null;
    final effectiveWeightsDType = weightsDType ?? DType.float32;
    final outDType = hasWeights ? DType.float64 : DType.int64;
    final weightLoadExpr = switch (effectiveWeightsDType) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => 'load_val_w(w_off)',
      DType.int32 ||
      DType.int16 ||
      DType.int8 ||
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean => 'f32(load_val_w(w_off))',
      DType.int64 => 'i64_to_f32(load_val_w(w_off))',
      DType.uint64 => 'u64_to_f32(load_val_w(w_off))',
      DType.complex64 || DType.complex128 => 'load_val_w(w_off).x',
    };
    final body = hasWeights
        ? '''
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < metadata.len_x; i = i + 1u) {
    let x_off = u32(i32(metadata.offset_x) + i32(i) * metadata.stride_x);
    let b = u32(load_x_idx(x_off));
    if (b == bin_idx) {
      let w_off = u32(i32(metadata.offset_w) + i32(i) * metadata.stride_w);
      acc = acc + ($weightLoadExpr);
    }
  }
  let dst_off = u32(i32(metadata.offset_out) + i32(bin_idx) * metadata.stride_out);
  store_raw_dst(dst_off, f32_to_f64(acc));
'''
        : '''
  var c: vec2<u32> = vec2<u32>(0u, 0u);
  for (var i: u32 = 0u; i < metadata.len_x; i = i + 1u) {
    let x_off = u32(i32(metadata.offset_x) + i32(i) * metadata.stride_x);
    let b = u32(load_x_idx(x_off));
    if (b == bin_idx) {
      c = u64_add(c, vec2<u32>(1u, 0u));
    }
  }
  let dst_off = u32(i32(metadata.offset_out) + i32(bin_idx) * metadata.stride_out);
  store_raw_dst(dst_off, c);
''';

    final dstBindingIndex = hasWeights ? 2 : 1;
    final metaBindingIndex = hasWeights ? 3 : 2;
    final weightDecls = hasWeights
        ? '''
${WgslDTypeCodec.readBindingDecl(1, 'w_arr', effectiveWeightsDType)}
'''
        : '';
    final weightHelpers = hasWeights
        ? '''
${WgslDTypeCodec.rawLoadFunction('load_raw_w', 'w_arr', effectiveWeightsDType)}
${WgslDTypeCodec.computeLoadFunction('load_val_w', 'load_raw_w', effectiveWeightsDType)}
'''
        : '';

    final code =
        '''
struct BincountMetadata {
  out_len: u32,
  len_x: u32,
  offset_x: u32,
  offset_w: u32,
  offset_out: u32,
  stride_x: i32,
  stride_w: i32,
  stride_out: i32,
};

${WgslDTypeCodec.readBindingDecl(0, 'x_arr', xDType)}
$weightDecls${WgslDTypeCodec.writeBindingDecl(dstBindingIndex, 'dst', outDType)}
@group(0) @binding($metaBindingIndex) var<uniform> metadata: BincountMetadata;

${WgslDTypeCodec.wgslNumericHelpers}
${WgslDTypeCodec.rawLoadFunction('load_raw_x', 'x_arr', xDType)}
${_indexLoadFunction('load_x_idx', 'load_raw_x', xDType)}
$weightHelpers${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', outDType)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let bin_idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (bin_idx >= metadata.out_len) {
    return;
  }
  $body
}
''';

    return _bincountCache[key] = WgslShaderModule(
      name: 'bincount_${xDType.name}_${weightsDType?.name ?? "unweighted"}',
      code: code,
      workgroupSize: WgslWorkgroupSize.linear1D,
      bindings: [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'x_arr',
          dtype: WgslDType.fromDType(xDType),
          access: WgslBufferAccess.read,
        ),
        if (hasWeights)
          WgslBinding(
            group: 0,
            binding: 1,
            name: 'w_arr',
            dtype: WgslDType.fromDType(effectiveWeightsDType),
            access: WgslBufferAccess.read,
          ),
        WgslBinding(
          group: 0,
          binding: dstBindingIndex,
          name: 'dst',
          dtype: WgslDType.fromDType(outDType),
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: metaBindingIndex,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'BincountMetadata',
        ),
      ],
    );
  }

  /// Generates a workgroup-parallel Hillis-Steele inclusive prefix scan shader
  /// (`cumsum` or `cumprod`) along an axis for [dtype].
  static WgslShaderModule cumulativeScanShader({
    required String op,
    required DType dtype,
  }) {
    final key = (op, dtype);
    if (_cumulativeScanCache[key] case final cached?) {
      return cached;
    }
    final isProd = op == 'cumprod';
    final valType = WgslDTypeCodec.computeValueType(dtype);
    final combineExpr = WgslDTypeCodec.binaryValueExpr(
      isProd ? 'mul' : 'add',
      dtype,
    );
    final identityExpr = switch (dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => isProd ? '1.0' : '0.0',
      DType.int32 || DType.int16 || DType.int8 => isProd ? '1' : '0',
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean => isProd ? '1u' : '0u',
      DType.int64 ||
      DType.uint64 => isProd ? 'vec2<u32>(1u, 0u)' : 'vec2<u32>(0u, 0u)',
      DType.complex64 || DType.complex128 =>
        isProd ? 'vec2<f32>(1.0, 0.0)' : 'vec2<f32>(0.0, 0.0)',
    };

    final code =
        '''
struct ScanMetadata {
  num_slices: u32,
  rank: u32,
  axis_size: u32,
  offset_src: u32,
  offset_dst: u32,
  stride_src_axis: i32,
  stride_dst_axis: i32,
  pad0: u32,
  slice_shape: array<vec4<u32>, 2>,
  strides_src: array<vec4<i32>, 2>,
  strides_dst: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: ScanMetadata;

${WgslDTypeCodec.wgslNumericHelpers}
${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.computeLoadFunction('load_val_src', 'load_raw_src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', dtype)}

var<workgroup> s_scan: array<$valType, 256>;
var<workgroup> s_carry: $valType;

fn combine_scan(a: $valType, b: $valType) -> $valType {
  return $combineExpr;
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>
) {
  let slice_idx = wg_id.y * num_wg.x + wg_id.x;
  let tid = local_id.x;
  let is_active = slice_idx < metadata.num_slices;

  var base_src: i32 = i32(metadata.offset_src);
  var base_dst: i32 = i32(metadata.offset_dst);

  if (is_active) {
    var rem = slice_idx;
    for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
      let d = metadata.rank - 1u - i;
      let dim_sz = metadata.slice_shape[d / 4u][d % 4u];
      let coord = i32(rem % dim_sz);
      rem = rem / dim_sz;
      base_src = base_src + coord * metadata.strides_src[d / 4u][d % 4u];
      base_dst = base_dst + coord * metadata.strides_dst[d / 4u][d % 4u];
    }
  }

  if (tid == 0u) {
    s_carry = $identityExpr;
  }
  workgroupBarrier();

  let n = metadata.axis_size;
  let num_blocks = (n + 255u) / 256u;
  for (var blk: u32 = 0u; blk < num_blocks; blk = blk + 1u) {
    let elem_idx = blk * 256u + tid;
    var val: $valType = $identityExpr;
    if (is_active && elem_idx < n) {
      let s_off = base_src + i32(elem_idx) * metadata.stride_src_axis;
      val = load_val_src(u32(s_off));
    }
    if (tid == 0u && blk > 0u) {
      val = combine_scan(s_carry, val);
    }
    s_scan[tid] = val;
    workgroupBarrier();

    var step_off: u32 = 1u;
    loop {
      if (step_off >= 256u) {
        break;
      }
      var merged = s_scan[tid];
      if (tid >= step_off) {
        merged = combine_scan(s_scan[tid - step_off], merged);
      }
      workgroupBarrier();
      s_scan[tid] = merged;
      workgroupBarrier();
      step_off = step_off << 1u;
    }

    if (is_active && elem_idx < n) {
      let d_off = base_dst + i32(elem_idx) * metadata.stride_dst_axis;
      store_val_dst(u32(d_off), s_scan[tid]);
    }
    if (tid == 255u) {
      s_carry = s_scan[255u];
    }
    workgroupBarrier();
  }
}
''';

    return _cumulativeScanCache[key] = WgslShaderModule(
      name: '${op}_${dtype.name}',
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
          customTypeName: 'ScanMetadata',
        ),
      ],
    );
  }

  /// Generates a parallel 1st-order difference stencil shader along an axis
  /// for [dtype].
  static WgslShaderModule diffShader(DType dtype) {
    if (_diffCache[dtype] case final cached?) {
      return cached;
    }
    final subExpr = WgslDTypeCodec.binaryValueExpr(
      dtype == DType.boolean ? 'bitwise_xor' : 'sub',
      dtype,
    );
    final code =
        '''
struct DiffMetadata {
  total_elements: u32,
  rank: u32,
  offset_src: u32,
  offset_dst: u32,
  stride_src_axis: i32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  shape_out: array<vec4<u32>, 2>,
  strides_src: array<vec4<i32>, 2>,
  strides_dst: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: DiffMetadata;

${WgslDTypeCodec.wgslNumericHelpers}
${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.computeLoadFunction('load_val_src', 'load_raw_src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}
${WgslDTypeCodec.computeStoreFunction('store_val_dst', 'store_raw_dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem = idx;
  var off_src: i32 = i32(metadata.offset_src);
  var off_dst: i32 = i32(metadata.offset_dst);

  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_sz = metadata.shape_out[d / 4u][d % 4u];
    let coord = i32(rem % dim_sz);
    rem = rem / dim_sz;
    off_src = off_src + coord * metadata.strides_src[d / 4u][d % 4u];
    off_dst = off_dst + coord * metadata.strides_dst[d / 4u][d % 4u];
  }

  let a = load_val_src(u32(off_src + metadata.stride_src_axis));
  let b = load_val_src(u32(off_src));
  store_val_dst(u32(off_dst), $subExpr);
}
''';

    return _diffCache[dtype] = WgslShaderModule(
      name: 'diff_${dtype.name}',
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
          customTypeName: 'DiffMetadata',
        ),
      ],
    );
  }
}
