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

/// Universal 15-[DType] WGSL compute shader generators for tensor manipulation
/// (`repeat`, `roll`, `diag`, `diagonal`, `triu`, `tril`) and GPU stream
/// compaction (`nonzero`, `flatnonzero`, `argwhere`, `extract`).
extension type const WgslManipulationKernels._(Object? _) {
  static final Map<DType, WgslShaderModule> _repeatCache = {};
  static final Map<DType, WgslShaderModule> _rollCache = {};
  static final Map<DType, WgslShaderModule> _diag1DCache = {};
  static final Map<DType, WgslShaderModule> _diagonalCache = {};
  static final Map<DType, WgslShaderModule> _triangularCache = {};
  static final Map<DType, WgslShaderModule> _nonZeroScanCache = {};
  static final Map<(DType, int), WgslShaderModule> _nonZeroScatterCache = {};

  /// Generates a universal `repeat` compute shader for [dtype].
  static WgslShaderModule repeatShader(DType dtype) {
    if (_repeatCache[dtype] case final cached?) {
      return cached;
    }
    final code =
        '''
struct RepeatMetadata {
  total_elements: u32,
  rank: u32,
  axis: u32,
  repeats: u32,
  offset_a: u32,
  offset_out: u32,
  pad0: u32,
  pad1: u32,
  shape_in: array<vec4<u32>, 2>,
  shape_out: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: RepeatMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  if (metadata.axis == 99u) {
    let off_out = metadata.offset_out + idx * u32(metadata.strides_out[0][0]);
    var rem_a = idx / metadata.repeats;
    var off_a: i32 = i32(metadata.offset_a);
    for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
      let d = metadata.rank - 1u - i;
      let dim_sz = metadata.shape_in[d / 4u][d % 4u];
      let coord = i32(rem_a % dim_sz);
      rem_a = rem_a / dim_sz;
      off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
    }
    store_raw_dst(off_out, load_raw_src(u32(off_a)));
    return;
  }

  var rem = idx;
  var off_a: i32 = i32(metadata.offset_a);
  var off_out: i32 = i32(metadata.offset_out);
  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let out_dim = metadata.shape_out[d / 4u][d % 4u];
    let coord_out = i32(rem % out_dim);
    rem = rem / out_dim;
    off_out = off_out + coord_out * metadata.strides_out[d / 4u][d % 4u];
    let coord_in = select(coord_out, coord_out / i32(metadata.repeats), d == metadata.axis);
    off_a = off_a + coord_in * metadata.strides_a[d / 4u][d % 4u];
  }
  store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
}
''';

    return _repeatCache[dtype] = WgslShaderModule(
      name: 'repeat_${dtype.name}',
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
          customTypeName: 'RepeatMetadata',
        ),
      ],
    );
  }

  /// Generates a universal `roll` compute shader for [dtype].
  static WgslShaderModule rollShader(DType dtype) {
    if (_rollCache[dtype] case final cached?) {
      return cached;
    }
    final code =
        '''
struct RollMetadata {
  total_elements: u32,
  rank: u32,
  axis: u32,
  shift: i32,
  offset_a: u32,
  offset_out: u32,
  pad0: u32,
  pad1: u32,
  shape: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: RollMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem_out = idx;
  var off_out: i32 = i32(metadata.offset_out);
  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_sz = metadata.shape[d / 4u][d % 4u];
    let coord = i32(rem_out % dim_sz);
    rem_out = rem_out / dim_sz;
    off_out = off_out + coord * metadata.strides_out[d / 4u][d % 4u];
  }

  if (metadata.axis == 99u) {
    let total_i = i32(metadata.total_elements);
    let src_flat = u32((((i32(idx) - metadata.shift) % total_i) + total_i) % total_i);
    var rem_a = src_flat;
    var off_a: i32 = i32(metadata.offset_a);
    for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
      let d = metadata.rank - 1u - i;
      let dim_sz = metadata.shape[d / 4u][d % 4u];
      let coord = i32(rem_a % dim_sz);
      rem_a = rem_a / dim_sz;
      off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
    }
    store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
    return;
  }

  var rem = idx;
  var off_a: i32 = i32(metadata.offset_a);
  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_sz = metadata.shape[d / 4u][d % 4u];
    var coord = i32(rem % dim_sz);
    rem = rem / dim_sz;
    if (d == metadata.axis) {
      let dim_i = i32(dim_sz);
      coord = (((coord - metadata.shift) % dim_i) + dim_i) % dim_i;
    }
    off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
  }
  store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
}
''';

    return _rollCache[dtype] = WgslShaderModule(
      name: 'roll_${dtype.name}',
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
          customTypeName: 'RollMetadata',
        ),
      ],
    );
  }

  /// Generates a 1D-to-2D diagonal construction (`diag`) shader for [dtype].
  static WgslShaderModule diag1DTo2DShader(DType dtype) {
    if (_diag1DCache[dtype] case final cached?) {
      return cached;
    }
    final zeroExpr = WgslDTypeCodec.rawZeroLiteral(dtype);
    final code =
        '''
struct Diag1DMetadata {
  mat_size: u32,
  vec_len: u32,
  k_offset: i32,
  offset_v: u32,
  stride_v: i32,
  offset_out: u32,
  stride_out_r: i32,
  stride_out_c: i32,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: Diag1DMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  let total = metadata.mat_size * metadata.mat_size;
  if (idx >= total) {
    return;
  }
  let r = i32(idx / metadata.mat_size);
  let c = i32(idx % metadata.mat_size);
  let off_out = u32(i32(metadata.offset_out) + r * metadata.stride_out_r + c * metadata.stride_out_c);
  if ((c - r) == metadata.k_offset) {
    let diag_i = select(r, c - metadata.k_offset, metadata.k_offset >= 0);
    if (diag_i >= 0 && diag_i < i32(metadata.vec_len)) {
      let off_v = u32(i32(metadata.offset_v) + diag_i * metadata.stride_v);
      store_raw_dst(off_out, load_raw_src(off_v));
      return;
    }
  }
  store_raw_dst(off_out, $zeroExpr);
}
''';

    return _diag1DCache[dtype] = WgslShaderModule(
      name: 'diag_1d_to_2d_${dtype.name}',
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
          customTypeName: 'Diag1DMetadata',
        ),
      ],
    );
  }

  /// Generates an N-D diagonal extraction (`diagonal`) shader for [dtype].
  static WgslShaderModule diagonalShader(DType dtype) {
    if (_diagonalCache[dtype] case final cached?) {
      return cached;
    }
    final code =
        '''
struct DiagonalMetadata {
  total_elements: u32,
  out_rank: u32,
  row_start: i32,
  col_start: i32,
  stride_ax1: i32,
  stride_ax2: i32,
  offset_a: u32,
  offset_out: u32,
  out_shape: array<vec4<u32>, 2>,
  rem_strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: DiagonalMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem = idx;
  var off_a: i32 = i32(metadata.offset_a);
  var off_out: i32 = i32(metadata.offset_out);
  var diag_step: i32 = 0;

  for (var i: u32 = 0u; i < metadata.out_rank; i = i + 1u) {
    let d = metadata.out_rank - 1u - i;
    let dim_sz = metadata.out_shape[d / 4u][d % 4u];
    let coord = i32(rem % dim_sz);
    rem = rem / dim_sz;
    off_out = off_out + coord * metadata.strides_out[d / 4u][d % 4u];
    if (d == (metadata.out_rank - 1u)) {
      diag_step = coord;
    } else {
      off_a = off_a + coord * metadata.rem_strides_a[d / 4u][d % 4u];
    }
  }

  off_a = off_a + (metadata.row_start + diag_step) * metadata.stride_ax1 + (metadata.col_start + diag_step) * metadata.stride_ax2;
  store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
}
''';

    return _diagonalCache[dtype] = WgslShaderModule(
      name: 'diagonal_${dtype.name}',
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
          customTypeName: 'DiagonalMetadata',
        ),
      ],
    );
  }

  /// Generates a universal `triu` / `tril` shader for [dtype].
  static WgslShaderModule triangularShader(DType dtype) {
    if (_triangularCache[dtype] case final cached?) {
      return cached;
    }
    final zeroExpr = WgslDTypeCodec.rawZeroLiteral(dtype);
    final code =
        '''
struct TriangularMetadata {
  total_elements: u32,
  rank: u32,
  k: i32,
  upper: u32,
  offset_a: u32,
  offset_out: u32,
  pad0: u32,
  pad1: u32,
  shape: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
  strides_out: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
${WgslDTypeCodec.writeBindingDecl(1, 'dst', dtype)}
@group(0) @binding(2) var<uniform> metadata: TriangularMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', dtype)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements) {
    return;
  }

  var rem = idx;
  var off_a: i32 = i32(metadata.offset_a);
  var off_out: i32 = i32(metadata.offset_out);
  var row: i32 = 0;
  var col: i32 = 0;

  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_size = metadata.shape[d / 4u][d % 4u];
    let coord = i32(rem % dim_size);
    rem = rem / dim_size;
    if (d == metadata.rank - 1u) {
      col = coord;
    } else if (d == metadata.rank - 2u) {
      row = coord;
    }
    off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
    off_out = off_out + coord * metadata.strides_out[d / 4u][d % 4u];
  }

  let keep = select(col <= (row + metadata.k), col >= (row + metadata.k), metadata.upper != 0u);
  if (keep) {
    store_raw_dst(u32(off_out), load_raw_src(u32(off_a)));
  } else {
    store_raw_dst(u32(off_out), $zeroExpr);
  }
}
''';

    return _triangularCache[dtype] = WgslShaderModule(
      name: 'triangular_${dtype.name}',
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
          customTypeName: 'TriangularMetadata',
        ),
      ],
    );
  }

  /// Generates Pass 1 of GPU stream compaction: evaluates non-zero predicate on
  /// strided input of [dtype] and computes exclusive prefix-sum offsets and
  /// total non-zero count in a single workgroup scan.
  static WgslShaderModule nonZeroScanShader(DType dtype) {
    if (_nonZeroScanCache[dtype] case final cached?) {
      return cached;
    }
    final isNonZeroExpr = switch (dtype) {
      DType.float64 => '((raw.y & 0x7FFFFFFFu) | raw.x) != 0u',
      DType.float32 => '(raw & 0x7FFFFFFFu) != 0u',
      DType.float16 || DType.bfloat16 => '(raw & 0x7FFFu) != 0u',
      DType.int64 || DType.uint64 => '(raw.x | raw.y) != 0u',
      DType.complex64 =>
        '((raw.x & 0x7FFFFFFFu) | (raw.y & 0x7FFFFFFFu)) != 0u',
      DType.complex128 =>
        '((raw.y & 0x7FFFFFFFu) | raw.x | (raw.w & 0x7FFFFFFFu) | raw.z) != 0u',
      DType.int32 ||
      DType.int16 ||
      DType.int8 ||
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean => 'raw != 0u',
    };

    final code =
        '''
struct ScanMetadata {
  total_elements: u32,
  rank: u32,
  offset_a: u32,
  pad0: u32,
  shape: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
@group(0) @binding(1) var<storage, read_write> prefix_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> total_count: array<u32>;
@group(0) @binding(3) var<uniform> metadata: ScanMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}

var<workgroup> s_scan: array<u32, 256>;
var<workgroup> s_carry: u32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  if (tid == 0u) {
    s_carry = 0u;
  }
  workgroupBarrier();

  let num_blocks = (metadata.total_elements + 255u) / 256u;
  for (var blk: u32 = 0u; blk < num_blocks; blk = blk + 1u) {
    let idx = blk * 256u + tid;
    var flag: u32 = 0u;
    if (idx < metadata.total_elements) {
      var rem = idx;
      var off_a: i32 = i32(metadata.offset_a);
      for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
        let d = metadata.rank - 1u - i;
        let dim_sz = metadata.shape[d / 4u][d % 4u];
        let coord = i32(rem % dim_sz);
        rem = rem / dim_sz;
        off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
      }
      let raw = load_raw_src(u32(off_a));
      flag = select(0u, 1u, $isNonZeroExpr);
    }

    s_scan[tid] = flag;
    workgroupBarrier();

    for (var step: u32 = 1u; step < 256u; step = step * 2u) {
      var val = s_scan[tid];
      if (tid >= step) {
        val = val + s_scan[tid - step];
      }
      workgroupBarrier();
      s_scan[tid] = val;
      workgroupBarrier();
    }

    let inclusive = s_scan[tid];
    let exclusive = inclusive - flag + s_carry;
    if (idx < metadata.total_elements) {
      prefix_offsets[idx] = (exclusive << 1u) | flag;
    }
    workgroupBarrier();

    if (tid == 255u) {
      s_carry = s_carry + inclusive;
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    total_count[0] = s_carry;
  }
}
''';

    return _nonZeroScanCache[dtype] = WgslShaderModule(
      name: 'nonzero_scan_${dtype.name}',
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
          name: 'prefix_offsets',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'total_count',
          dtype: WgslDType.uint32,
          access: WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'metadata',
          isUniform: true,
          customTypeName: 'ScanMetadata',
        ),
      ],
    );
  }

  /// Generates Pass 2 of GPU stream compaction: scatters compacted flat indices
  /// (`mode == 0`), N-D coordinates (`mode == 1`), single-axis coordinates
  /// (`mode == 2`), or extracted elements (`mode == 3`).
  static WgslShaderModule nonZeroScatterShader({
    required DType dtype,
    required int mode,
  }) {
    final key = (dtype, mode);
    if (_nonZeroScatterCache[key] case final cached?) {
      return cached;
    }
    final outDType = mode == 3 ? dtype : DType.int64;
    final String body;
    if (mode == 0) {
      body =
          'store_raw_dst(metadata.offset_out + out_pos, vec2<u32>(idx, 0u));';
    } else if (mode == 1) {
      body = '''
  var rem = idx;
  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_sz = metadata.shape[d / 4u][d % 4u];
    let coord = rem % dim_sz;
    rem = rem / dim_sz;
    store_raw_dst(metadata.offset_out + out_pos * metadata.rank + d, vec2<u32>(coord, 0u));
  }''';
    } else if (mode == 2) {
      body = '''
  var rem = idx;
  var target_coord: u32 = 0u;
  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_sz = metadata.shape[d / 4u][d % 4u];
    let coord = rem % dim_sz;
    rem = rem / dim_sz;
    if (d == metadata.target_axis) {
      target_coord = coord;
    }
  }
  store_raw_dst(metadata.offset_out + out_pos, vec2<u32>(target_coord, 0u));''';
    } else {
      body = '''
  var rem = idx;
  var off_a: i32 = i32(metadata.offset_a);
  for (var i: u32 = 0u; i < metadata.rank; i = i + 1u) {
    let d = metadata.rank - 1u - i;
    let dim_sz = metadata.shape[d / 4u][d % 4u];
    let coord = i32(rem % dim_sz);
    rem = rem / dim_sz;
    off_a = off_a + coord * metadata.strides_a[d / 4u][d % 4u];
  }
  store_raw_dst(metadata.offset_out + out_pos, load_raw_src(u32(off_a)));''';
    }

    final code =
        '''
struct ScatterMetadata {
  total_elements: u32,
  rank: u32,
  offset_a: u32,
  target_axis: u32,
  offset_out: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  shape: array<vec4<u32>, 2>,
  strides_a: array<vec4<i32>, 2>,
};

${WgslDTypeCodec.readBindingDecl(0, 'src', dtype)}
@group(0) @binding(1) var<storage, read> prefix_offsets: array<u32>;
${WgslDTypeCodec.writeBindingDecl(2, 'dst', outDType)}
@group(0) @binding(3) var<uniform> metadata: ScatterMetadata;

${WgslDTypeCodec.rawLoadFunction('load_raw_src', 'src', dtype)}
${WgslDTypeCodec.rawStoreFunction('store_raw_dst', 'dst', outDType)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let idx = global_id.y * (num_wg.x * 256u) + global_id.x;
  if (idx >= metadata.total_elements || arrayLength(&src) == 0u) {
    return;
  }
  let packed = prefix_offsets[idx];
  if ((packed & 1u) == 0u) {
    return;
  }
  let out_pos = packed >> 1u;
  $body
}
''';

    return _nonZeroScatterCache[key] = WgslShaderModule(
      name: 'nonzero_scatter_${mode}_${dtype.name}',
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
          name: 'prefix_offsets',
          dtype: WgslDType.uint32,
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
          customTypeName: 'ScatterMetadata',
        ),
      ],
    );
  }
}
