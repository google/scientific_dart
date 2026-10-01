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
import 'dart:typed_data';

import '../buffer.dart';
import '../device.dart';
import 'linalg_wgsl_df64.dart';

const String _detAndSlogdetShader =
    '''
$linalgDf64WgslLibrary

struct DetParams {
  n: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> work_lu: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_det: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_sign: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out_logabsdet: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> params: DetParams;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  let one = df64_one();
  if (n == 0u) {
    out_det[0] = pack_df64_f64(one);
    out_sign[0] = pack_df64_f64(one);
    out_logabsdet[0] = vec2<u32>(0u, 0u);
    return;
  }

  for (var i = 0u; i < n * n; i = i + 1u) {
    work_lu[i] = in_a[i];
  }

  var sign_val = 1.0;
  for (var k = 0u; k < n; k = k + 1u) {
    var pivot_row = k;
    var max_val = df64_abs(unpack_f64_df64(work_lu[k * n + k]));
    for (var r = k + 1u; r < n; r = r + 1u) {
      let cand = df64_abs(unpack_f64_df64(work_lu[r * n + k]));
      if (df64_gt(cand, max_val)) {
        max_val = cand;
        pivot_row = r;
      }
    }
    if (pivot_row != k) {
      sign_val = -sign_val;
      for (var c = 0u; c < n; c = c + 1u) {
        let tmp = work_lu[k * n + c];
        work_lu[k * n + c] = work_lu[pivot_row * n + c];
        work_lu[pivot_row * n + c] = tmp;
      }
    }
    let piv = unpack_f64_df64(work_lu[k * n + k]);
    if (abs(piv.x) + abs(piv.y) <= 1e-15) {
      out_det[0] = vec2<u32>(0u, 0u);
      out_sign[0] = vec2<u32>(0u, 0u);
      // -Infinity in IEEE-754 f64
      out_logabsdet[0] = vec2<u32>(0u, 0xFFF00000u);
      return;
    }
    for (var i = k + 1u; i < n; i = i + 1u) {
      let mult = df64_div(unpack_f64_df64(work_lu[i * n + k]), piv);
      for (var j = k + 1u; j < n; j = j + 1u) {
        let u_kj = unpack_f64_df64(work_lu[k * n + j]);
        let a_ij = unpack_f64_df64(work_lu[i * n + j]);
        work_lu[i * n + j] = pack_df64_f64(df64_sub(a_ij, df64_mul(mult, u_kj)));
      }
    }
  }

  var det_acc = df64_from_f32(sign_val);
  var log_acc = df64_zero();
  for (var i = 0u; i < n; i = i + 1u) {
    let diag = unpack_f64_df64(work_lu[i * n + i]);
    det_acc = df64_mul(det_acc, diag);
    if (diag.x < 0.0 || (diag.x == 0.0 && diag.y < 0.0)) {
      sign_val = -sign_val;
    }
    log_acc = df64_add(log_acc, df64_log(df64_abs(diag)));
  }
  out_det[0] = pack_df64_f64(det_acc);
  out_sign[0] = pack_df64_f64(df64_from_f32(sign_val));
  out_logabsdet[0] = pack_df64_f64(log_acc);
}
''';

const String _identityF64Shader =
    '''
$linalgDf64WgslLibrary

struct IdentityParams {
  n: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> out_eye: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> params: IdentityParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let n = params.n;
  if (idx >= n * n) { return; }
  let r = idx / n;
  let c = idx % n;
  out_eye[idx] = select(vec2<u32>(0u, 0u), vec2<u32>(0u, 0x3FF00000u), r == c);
}
''';

const String _pinvFromSvdShader =
    '''
$linalgDf64WgslLibrary

struct PinvParams {
  m: u32,
  n: u32,
  nrhs: u32,
  rcond_hi_bits: u32,
  rcond_lo_bits: u32,
  compute_lstsq: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_u: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_s: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> in_vt: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_pinv: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out_rank: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> params: PinvParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let k_min = min(m, n);
  let rcond = vec2<f32>(bitcast<f32>(params.rcond_hi_bits), bitcast<f32>(params.rcond_lo_bits));
  let s0 = select(df64_zero(), unpack_f64_df64(in_s[0]), k_min > 0u);
  let cutoff = df64_mul(rcond, s0);

  var rank_count = 0u;
  for (var r = 0u; r < k_min; r = r + 1u) {
    let sr = unpack_f64_df64(in_s[r]);
    if (df64_gt(sr, cutoff)) {
      rank_count = rank_count + 1u;
    }
  }
  out_rank[0] = pack_df64_f64(df64_from_f32(f32(rank_count)));

  // Compute pinv (n x m): pinv[i, j] = sum_{r=0..k_min-1} V[i, r] * (1/S[r]) * U[j, r]
  // Note: V[i, r] = Vt[r, i].
  let one = df64_one();
  for (var i = 0u; i < n; i = i + 1u) {
    for (var j = 0u; j < m; j = j + 1u) {
      var acc = df64_zero();
      for (var r = 0u; r < k_min; r = r + 1u) {
        let sr = unpack_f64_df64(in_s[r]);
        if (df64_gt(sr, cutoff)) {
          let inv_s = df64_div(one, sr);
          let v_ir = unpack_f64_df64(in_vt[r * n + i]);
          let u_jr = unpack_f64_df64(in_u[j * k_min + r]);
          acc = df64_add(acc, df64_mul(df64_mul(v_ir, inv_s), u_jr));
        }
      }
      out_pinv[i * m + j] = pack_df64_f64(acc);
    }
  }
}
''';

const String _lstsqFromPinvShader =
    '''
$linalgDf64WgslLibrary

struct PinvParams {
  m: u32,
  n: u32,
  nrhs: u32,
  rcond_hi_bits: u32,
  rcond_lo_bits: u32,
  compute_lstsq: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_pinv: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> in_b: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_sol: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out_res: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> params: PinvParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let nrhs = params.nrhs;
  // x (n x nrhs) = pinv (n x m) * b (m x nrhs)
  for (var i = 0u; i < n; i = i + 1u) {
    for (var c = 0u; c < nrhs; c = c + 1u) {
      var acc = df64_zero();
      for (var j = 0u; j < m; j = j + 1u) {
        let p_ij = unpack_f64_df64(in_pinv[i * m + j]);
        let b_jc = unpack_f64_df64(in_b[j * nrhs + c]);
        acc = df64_add(acc, df64_mul(p_ij, b_jc));
      }
      out_sol[i * nrhs + c] = pack_df64_f64(acc);
    }
  }
  // Residuals per RHS column: sum_{i=0..m-1} (b[i, c] - (A * x)[i, c])^2
  for (var c = 0u; c < nrhs; c = c + 1u) {
    var res_sum = df64_zero();
    for (var i = 0u; i < m; i = i + 1u) {
      var ax_ic = df64_zero();
      for (var j = 0u; j < n; j = j + 1u) {
        let a_ij = unpack_f64_df64(in_a[i * n + j]);
        let x_jc = unpack_f64_df64(out_sol[j * nrhs + c]);
        ax_ic = df64_add(ax_ic, df64_mul(a_ij, x_jc));
      }
      let diff = df64_sub(unpack_f64_df64(in_b[i * nrhs + c]), ax_ic);
      res_sum = df64_add(res_sum, df64_mul(diff, diff));
    }
    out_res[c] = pack_df64_f64(res_sum);
  }
}
''';

const String _vectorAndMatrixNormShader =
    '''
$linalgDf64WgslLibrary

struct NormParams {
  outer_size: u32,
  axis_len: u32,
  inner_size: u32,
  norm_mode: u32,
  p_hi_bits: u32,
  p_lo_bits: u32,
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out_norm: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: NormParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_out = params.outer_size * params.inner_size;
  if (idx >= total_out) { return; }

  let mode = params.norm_mode;
  let one = df64_one();

  // Matrix norms (modes 10..13): single output scalar (idx == 0).
  if (mode >= 10u) {
    let rows = params.rows;
    let cols = params.cols;
    if (mode == 10u || mode == 11u) {
      // mode 10: 1-norm (max col sum), mode 11: -1-norm (min col sum)
      var best = df64_zero();
      for (var c = 0u; c < cols; c = c + 1u) {
        var col_sum = df64_zero();
        for (var r = 0u; r < rows; r = r + 1u) {
          col_sum = df64_add(col_sum, df64_abs(unpack_f64_df64(in_a[r * cols + c])));
        }
        if (c == 0u) {
          best = col_sum;
        } else if (mode == 10u) {
          best = df64_max(best, col_sum);
        } else {
          best = df64_min(best, col_sum);
        }
      }
      out_norm[0] = pack_df64_f64(best);
      return;
    }
    if (mode == 12u || mode == 13u) {
      // mode 12: inf-norm (max row sum), mode 13: -inf-norm (min row sum)
      var best = df64_zero();
      for (var r = 0u; r < rows; r = r + 1u) {
        var row_sum = df64_zero();
        for (var c = 0u; c < cols; c = c + 1u) {
          row_sum = df64_add(row_sum, df64_abs(unpack_f64_df64(in_a[r * cols + c])));
        }
        if (r == 0u) {
          best = row_sum;
        } else if (mode == 12u) {
          best = df64_max(best, row_sum);
        } else {
          best = df64_min(best, row_sum);
        }
      }
      out_norm[0] = pack_df64_f64(best);
      return;
    }
    if (mode == 14u) {
      // mode 14: nuclear norm = sum of singular values in in_a[0..axis_len]
      var sum_s = df64_zero();
      for (var i = 0u; i < params.axis_len; i = i + 1u) {
        sum_s = df64_add(sum_s, unpack_f64_df64(in_a[i]));
      }
      out_norm[0] = pack_df64_f64(sum_s);
      return;
    }
    if (mode == 15u) {
      // mode 15: 2-norm = largest singular value in_a[0]
      out_norm[0] = in_a[0];
      return;
    }
    if (mode == 16u) {
      // mode 16: -2-norm = smallest singular value in_a[axis_len - 1]
      out_norm[0] = in_a[params.axis_len - 1u];
      return;
    }
    if (mode == 17u) {
      // mode 17: cond(2) = in_a[0] / in_a[axis_len - 1]
      let s_max = unpack_f64_df64(in_a[0]);
      let s_min = unpack_f64_df64(in_a[params.axis_len - 1u]);
      out_norm[0] = pack_df64_f64(df64_div(s_max, s_min));
      return;
    }
    if (mode == 18u) {
      // mode 18: cond(-2) = in_a[axis_len - 1] / in_a[0]
      let s_max = unpack_f64_df64(in_a[0]);
      let s_min = unpack_f64_df64(in_a[params.axis_len - 1u]);
      out_norm[0] = pack_df64_f64(df64_div(s_min, s_max));
      return;
    }
  }

  let outer = idx / params.inner_size;
  let inner = idx % params.inner_size;
  let n_len = params.axis_len;

  if (mode == 0u) {
    // 2-norm / Frobenius norm
    var sum_sq = df64_zero();
    for (var k = 0u; k < n_len; k = k + 1u) {
      let pos = (outer * n_len + k) * params.inner_size + inner;
      let v = unpack_f64_df64(in_a[pos]);
      sum_sq = df64_add(sum_sq, df64_mul(v, v));
    }
    out_norm[idx] = pack_df64_f64(df64_sqrt(sum_sq));
  } else if (mode == 1u) {
    // 1-norm
    var sum_abs = df64_zero();
    for (var k = 0u; k < n_len; k = k + 1u) {
      let pos = (outer * n_len + k) * params.inner_size + inner;
      let v = df64_abs(unpack_f64_df64(in_a[pos]));
      sum_abs = df64_add(sum_abs, v);
    }
    out_norm[idx] = pack_df64_f64(sum_abs);
  } else if (mode == 2u) {
    // inf-norm
    var max_v = df64_zero();
    for (var k = 0u; k < n_len; k = k + 1u) {
      let pos = (outer * n_len + k) * params.inner_size + inner;
      let v = df64_abs(unpack_f64_df64(in_a[pos]));
      if (k == 0u) {
        max_v = v;
      } else {
        max_v = df64_max(max_v, v);
      }
    }
    out_norm[idx] = pack_df64_f64(max_v);
  } else if (mode == 3u) {
    // -inf-norm
    var min_v = df64_zero();
    for (var k = 0u; k < n_len; k = k + 1u) {
      let pos = (outer * n_len + k) * params.inner_size + inner;
      let v = df64_abs(unpack_f64_df64(in_a[pos]));
      if (k == 0u) {
        min_v = v;
      } else {
        min_v = df64_min(min_v, v);
      }
    }
    out_norm[idx] = pack_df64_f64(min_v);
  } else if (mode == 4u) {
    // 0-norm (count non-zero)
    var nz = 0u;
    for (var k = 0u; k < n_len; k = k + 1u) {
      let pos = (outer * n_len + k) * params.inner_size + inner;
      let v = unpack_f64_df64(in_a[pos]);
      if (v.x != 0.0 || v.y != 0.0) {
        nz = nz + 1u;
      }
    }
    out_norm[idx] = pack_df64_f64(df64_from_f32(f32(nz)));
  } else {
    // General p-norm
    let p_val = bitcast<f32>(params.p_hi_bits);
    var acc = 0.0;
    for (var k = 0u; k < n_len; k = k + 1u) {
      let pos = (outer * n_len + k) * params.inner_size + inner;
      let v = unpack_f64_df64(in_a[pos]);
      let av = abs(v.x + v.y);
      if (av > 0.0) {
        acc = acc + pow(av, p_val);
      }
    }
    let res = select(0.0, pow(acc, 1.0 / p_val), acc > 0.0);
    out_norm[idx] = pack_df64_f64(df64_from_f32(res));
  }
}
''';

const String _scalarMulF64Shader =
    '''
$linalgDf64WgslLibrary

@group(0) @binding(0) var<storage, read> in_x: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_y: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_z: array<vec2<u32>>;

@compute @workgroup_size(1)
fn main() {
  let x = unpack_f64_df64(in_x[0]);
  let y = unpack_f64_df64(in_y[0]);
  out_z[0] = pack_df64_f64(df64_mul(x, y));
}
''';

/// Dispatches the determinant and log-determinant kernel on [device].
({GpuBuffer det, GpuBuffer sign, GpuBuffer logabsdet}) dispatchDetAndSlogdetGpu(
  GpuDevice device,
  GpuBuffer inputF64,
  int n,
) {
  final workLu = device.createBuffer(sizeInBytes: math.max(1, n * n) * 8);
  final outDet = device.createBuffer(sizeInBytes: 8);
  final outSign = device.createBuffer(sizeInBytes: 8);
  final outLogAbsDet = device.createBuffer(sizeInBytes: 8);

  final module = getOrCreateLinalgShader(
    'linalg_det_slogdet_f64',
    () => _detAndSlogdetShader,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputF64, workLu, outDet, outSign, outLogAbsDet],
    uniforms: [n, 0, 0, 0],
    workgroupsX: 1,
  );
  return (det: outDet, sign: outSign, logabsdet: outLogAbsDet);
}

/// Dispatches an `n x n` identity matrix `Float64` buffer kernel on [device].
GpuBuffer dispatchIdentityF64Gpu(GpuDevice device, int n) {
  final count = n * n;
  final outEye = device.createBuffer(sizeInBytes: math.max(1, count) * 8);
  if (count == 0) return outEye;

  final module = getOrCreateLinalgShader(
    'linalg_identity_f64',
    () => _identityF64Shader,
    workgroupSize: 64,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [outEye],
    uniforms: [n, 0, 0, 0],
    workgroupsX: (count + 63) ~/ 64,
  );
  return outEye;
}

/// Dispatches the pseudo-inverse and least-squares kernel from SVD factors.
({GpuBuffer pinv, GpuBuffer solution, GpuBuffer residuals, GpuBuffer rank})
dispatchPinvAndLstsqFromSvdGpu(
  GpuDevice device, {
  required GpuBuffer aF64,
  required GpuBuffer uF64,
  required GpuBuffer sF64,
  required GpuBuffer vtF64,
  required GpuBuffer bF64,
  required int m,
  required int n,
  required int nrhs,
  required double rcond,
  required bool computeLstsq,
}) {
  final outPinv = device.createBuffer(sizeInBytes: math.max(1, n * m) * 8);
  final outSol = device.createBuffer(sizeInBytes: math.max(1, n * nrhs) * 8);
  final outRes = device.createBuffer(sizeInBytes: math.max(1, nrhs) * 8);
  final outRank = device.createBuffer(sizeInBytes: 8);

  final hi = rcond;
  final lo = rcond - hi;
  final bd = ByteData(8)
    ..setFloat32(0, hi, Endian.little)
    ..setFloat32(4, lo, Endian.little);
  final rcondHiBits = bd.getUint32(0, Endian.little);
  final rcondLoBits = bd.getUint32(4, Endian.little);

  final uniforms = [
    m,
    n,
    nrhs,
    rcondHiBits,
    rcondLoBits,
    computeLstsq ? 1 : 0,
    0,
    0,
  ];

  final pinvModule = getOrCreateLinalgShader(
    'linalg_pinv_f64',
    () => _pinvFromSvdShader,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: pinvModule,
    buffers: [uF64, sF64, vtF64, outPinv, outRank],
    uniforms: uniforms,
    workgroupsX: 1,
  );

  if (computeLstsq) {
    final lstsqModule = getOrCreateLinalgShader(
      'linalg_lstsq_f64',
      () => _lstsqFromPinvShader,
    );
    device.backend.dispatchComputePipeline(
      shaderModule: lstsqModule,
      buffers: [aF64, outPinv, bF64, outSol, outRes],
      uniforms: uniforms,
      workgroupsX: 1,
    );
  }
  return (pinv: outPinv, solution: outSol, residuals: outRes, rank: outRank);
}

/// Dispatches the vector/matrix norm reduction kernel on [device].
GpuBuffer dispatchNormGpu(
  GpuDevice device,
  GpuBuffer inputF64, {
  required int outerSize,
  required int axisLength,
  required int innerSize,
  required int normMode,
  double pValue = 2.0,
  int rows = 0,
  int cols = 0,
}) {
  final totalOut = math.max(1, outerSize * innerSize);
  final outNorm = device.createBuffer(sizeInBytes: totalOut * 8);

  final bd = ByteData(8)..setFloat32(0, pValue, Endian.little);
  final pHiBits = bd.getUint32(0, Endian.little);

  final module = getOrCreateLinalgShader(
    'linalg_norm_f64',
    () => _vectorAndMatrixNormShader,
    workgroupSize: 64,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputF64, outNorm],
    uniforms: [
      outerSize,
      axisLength,
      innerSize,
      normMode,
      pHiBits,
      0,
      rows,
      cols,
    ],
    workgroupsX: (totalOut + 63) ~/ 64,
  );
  return outNorm;
}

/// Multiplies two 0D scalar `Float64` buffers on [device].
GpuBuffer dispatchScalarMulF64Gpu(
  GpuDevice device,
  GpuBuffer xF64,
  GpuBuffer yF64,
) {
  final outZ = device.createBuffer(sizeInBytes: 8);
  final module = getOrCreateLinalgShader(
    'linalg_scalar_mul_f64',
    () => _scalarMulF64Shader,
  );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [xF64, yF64, outZ],
    uniforms: const [],
    workgroupsX: 1,
  );
  return outZ;
}
