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

const String _choleskyRealShader =
    '''
$linalgDf64WgslLibrary

struct CholParams {
  n: u32,
  upper: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out_factor: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: CholParams;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  for (var idx = 0u; idx < n * n; idx = idx + 1u) {
    out_factor[idx] = vec2<u32>(0u, 0u);
  }
  for (var i = 0u; i < n; i = i + 1u) {
    for (var j = 0u; j <= i; j = j + 1u) {
      var sum = unpack_f64_df64(in_a[i * n + j]);
      for (var k = 0u; k < j; k = k + 1u) {
        let lik = unpack_f64_df64(out_factor[i * n + k]);
        let ljk = unpack_f64_df64(out_factor[j * n + k]);
        sum = df64_sub(sum, df64_mul(lik, ljk));
      }
      if (i == j) {
        out_factor[i * n + i] = pack_df64_f64(df64_sqrt(sum));
      } else {
        let ljj = unpack_f64_df64(out_factor[j * n + j]);
        out_factor[i * n + j] = pack_df64_f64(df64_div(sum, ljj));
      }
    }
  }
  if (params.upper != 0u) {
    for (var i = 0u; i < n; i = i + 1u) {
      for (var j = i + 1u; j < n; j = j + 1u) {
        out_factor[i * n + j] = out_factor[j * n + i];
        out_factor[j * n + i] = vec2<u32>(0u, 0u);
      }
    }
  }
}
''';

const String _qrRealShader =
    '''
$linalgDf64WgslLibrary

struct QrParams {
  m: u32,
  n: u32,
  q_cols: u32,
  r_rows: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> work_q: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> work_r: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_q: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out_r: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> params: QrParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let k_min = min(m, n);

  for (var i = 0u; i < m * n; i = i + 1u) {
    work_r[i] = in_a[i];
  }
  for (var r = 0u; r < m; r = r + 1u) {
    for (var c = 0u; c < m; c = c + 1u) {
      work_q[r * m + c] = select(vec2<u32>(0u, 0u), pack_df64_f64(df64_one()), r == c);
    }
  }

  for (var k = 0u; k < k_min; k = k + 1u) {
    var norm_sq = df64_zero();
    for (var i = k; i < m; i = i + 1u) {
      let val = unpack_f64_df64(work_r[i * n + k]);
      norm_sq = df64_add(norm_sq, df64_mul(val, val));
    }
    let norm_x = df64_sqrt(norm_sq);
    if (norm_x.x > 0.0 || norm_x.y > 0.0) {
      let x0 = unpack_f64_df64(work_r[k * n + k]);
      let alpha = select(norm_x, df64_neg(norm_x), x0.x >= 0.0);
      let v0 = df64_sub(x0, alpha);
      var tail_sq = df64_zero();
      for (var i = k + 1u; i < m; i = i + 1u) {
        let vi = unpack_f64_df64(work_r[i * n + k]);
        tail_sq = df64_add(tail_sq, df64_mul(vi, vi));
      }
      let v_norm_sq = df64_add(df64_mul(v0, v0), tail_sq);
      if (v_norm_sq.x > 0.0 || v_norm_sq.y > 0.0) {
        // Temporarily store v0 at work_r[k * n + k] to apply reflector cleanly.
        work_r[k * n + k] = pack_df64_f64(v0);
        let two = df64_from_f32(2.0);
        // Apply reflector from the left to columns k + 1 .. n - 1 of R.
        for (var c = k + 1u; c < n; c = c + 1u) {
          var dot_rc = df64_zero();
          for (var i = k; i < m; i = i + 1u) {
            let vi = unpack_f64_df64(work_r[i * n + k]);
            let ric = unpack_f64_df64(work_r[i * n + c]);
            dot_rc = df64_add(dot_rc, df64_mul(vi, ric));
          }
          let scale = df64_div(df64_mul(two, dot_rc), v_norm_sq);
          for (var i = k; i < m; i = i + 1u) {
            let vi = unpack_f64_df64(work_r[i * n + k]);
            let ric = unpack_f64_df64(work_r[i * n + c]);
            work_r[i * n + c] = pack_df64_f64(df64_sub(ric, df64_mul(scale, vi)));
          }
        }
        // Apply reflector from the right to all rows 0 .. m - 1 of Q.
        for (var r = 0u; r < m; r = r + 1u) {
          var dot_qr = df64_zero();
          for (var i = k; i < m; i = i + 1u) {
            let qri = unpack_f64_df64(work_q[r * m + i]);
            let vi = unpack_f64_df64(work_r[i * n + k]);
            dot_qr = df64_add(dot_qr, df64_mul(qri, vi));
          }
          let scale_q = df64_div(df64_mul(two, dot_qr), v_norm_sq);
          for (var i = k; i < m; i = i + 1u) {
            let qri = unpack_f64_df64(work_q[r * m + i]);
            let vi = unpack_f64_df64(work_r[i * n + k]);
            work_q[r * m + i] = pack_df64_f64(df64_sub(qri, df64_mul(scale_q, vi)));
          }
        }
        // Set diagonal of R to alpha and zero below diagonal.
        work_r[k * n + k] = pack_df64_f64(alpha);
        for (var i = k + 1u; i < m; i = i + 1u) {
          work_r[i * n + k] = vec2<u32>(0u, 0u);
        }
      }
    }
  }

  let q_cols = params.q_cols;
  for (var r = 0u; r < m; r = r + 1u) {
    for (var c = 0u; c < q_cols; c = c + 1u) {
      out_q[r * q_cols + c] = work_q[r * m + c];
    }
  }
  let r_rows = params.r_rows;
  for (var r = 0u; r < r_rows; r = r + 1u) {
    for (var c = 0u; c < n; c = c + 1u) {
      out_r[r * n + c] = work_r[r * n + c];
    }
  }
}
''';

const String _luDecomposeShader =
    '''
$linalgDf64WgslLibrary

struct LuParams {
  m: u32,
  n: u32,
  k_min: u32,
  pad: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> work_lu: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_p: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_l: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out_u: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> out_pivots: array<vec2<u32>>;
@group(0) @binding(6) var<uniform> params: LuParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let k_min = params.k_min;

  for (var i = 0u; i < m * n; i = i + 1u) {
    work_lu[i] = in_a[i];
  }

  // Track row permutation via out_p: initialize out_p to identity m x m.
  for (var r = 0u; r < m; r = r + 1u) {
    for (var c = 0u; c < m; c = c + 1u) {
      out_p[r * m + c] = select(vec2<u32>(0u, 0u), pack_df64_f64(df64_one()), r == c);
    }
  }

  for (var k = 0u; k < k_min; k = k + 1u) {
    var pivot_row = k;
    var max_val = df64_abs(unpack_f64_df64(work_lu[k * n + k]));
    for (var r = k + 1u; r < m; r = r + 1u) {
      let cand = df64_abs(unpack_f64_df64(work_lu[r * n + k]));
      if (df64_gt(cand, max_val)) {
        max_val = cand;
        pivot_row = r;
      }
    }
    out_pivots[k] = pack_df64_f64(df64_from_f32(f32(pivot_row)));

    if (pivot_row != k) {
      for (var c = 0u; c < n; c = c + 1u) {
        let tmp_lu = work_lu[k * n + c];
        work_lu[k * n + c] = work_lu[pivot_row * n + c];
        work_lu[pivot_row * n + c] = tmp_lu;
      }
      // Swap columns k and pivot_row of P so P * L * U = A.
      for (var r = 0u; r < m; r = r + 1u) {
        let tmp_p = out_p[r * m + k];
        out_p[r * m + k] = out_p[r * m + pivot_row];
        out_p[r * m + pivot_row] = tmp_p;
      }
    }

    let piv = unpack_f64_df64(work_lu[k * n + k]);
    if (piv.x != 0.0 || piv.y != 0.0) {
      for (var i = k + 1u; i < m; i = i + 1u) {
        let mult = df64_div(unpack_f64_df64(work_lu[i * n + k]), piv);
        work_lu[i * n + k] = pack_df64_f64(mult);
        for (var j = k + 1u; j < n; j = j + 1u) {
          let u_kj = unpack_f64_df64(work_lu[k * n + j]);
          let a_ij = unpack_f64_df64(work_lu[i * n + j]);
          work_lu[i * n + j] = pack_df64_f64(df64_sub(a_ij, df64_mul(mult, u_kj)));
        }
      }
    }
  }

  // Extract L (m x k_min) and U (k_min x n).
  let one_bits = pack_df64_f64(df64_one());
  for (var i = 0u; i < m; i = i + 1u) {
    for (var j = 0u; j < k_min; j = j + 1u) {
      if (i > j) {
        out_l[i * k_min + j] = work_lu[i * n + j];
      } else if (i == j) {
        out_l[i * k_min + j] = one_bits;
      } else {
        out_l[i * k_min + j] = vec2<u32>(0u, 0u);
      }
    }
  }
  for (var i = 0u; i < k_min; i = i + 1u) {
    for (var j = 0u; j < n; j = j + 1u) {
      if (j >= i) {
        out_u[i * n + j] = work_lu[i * n + j];
      } else {
        out_u[i * n + j] = vec2<u32>(0u, 0u);
      }
    }
  }
}
''';

const String _luSolveShader =
    '''
$linalgDf64WgslLibrary

struct LuSolveParams {
  n: u32,
  nrhs: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_lu: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> in_pivots: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> in_b: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_x: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params: LuSolveParams;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  let nrhs = params.nrhs;

  for (var i = 0u; i < n * nrhs; i = i + 1u) {
    out_x[i] = in_b[i];
  }

  for (var k = 0u; k < n; k = k + 1u) {
    let piv_f = unpack_f64_df64(in_pivots[k]);
    let piv = u32(max(0.0, round(piv_f.x + piv_f.y)));
    if (piv != k && piv < n) {
      for (var c = 0u; c < nrhs; c = c + 1u) {
        let tmp = out_x[k * nrhs + c];
        out_x[k * nrhs + c] = out_x[piv * nrhs + c];
        out_x[piv * nrhs + c] = tmp;
      }
    }
  }

  // Forward substitution: L * y = P * b (unit lower triangular).
  for (var c = 0u; c < nrhs; c = c + 1u) {
    for (var i = 0u; i < n; i = i + 1u) {
      var sum = unpack_f64_df64(out_x[i * nrhs + c]);
      for (var j = 0u; j < i; j = j + 1u) {
        let lij = unpack_f64_df64(in_lu[i * n + j]);
        let xjc = unpack_f64_df64(out_x[j * nrhs + c]);
        sum = df64_sub(sum, df64_mul(lij, xjc));
      }
      out_x[i * nrhs + c] = pack_df64_f64(sum);
    }

    // Back substitution: U * x = y.
    if (n > 0u) {
      var i = i32(n) - 1;
      loop {
        if (i < 0) { break; }
        let ui = u32(i);
        var sum = unpack_f64_df64(out_x[ui * nrhs + c]);
        for (var j = ui + 1u; j < n; j = j + 1u) {
          let uij = unpack_f64_df64(in_lu[ui * n + j]);
          let xjc = unpack_f64_df64(out_x[j * nrhs + c]);
          sum = df64_sub(sum, df64_mul(uij, xjc));
        }
        let uii = unpack_f64_df64(in_lu[ui * n + ui]);
        out_x[ui * nrhs + c] = pack_df64_f64(df64_div(sum, uii));
        i = i - 1;
      }
    }
  }
}
''';

const String _eighRealShader =
    '''
$linalgDf64WgslLibrary

struct EighParams {
  n: u32,
  use_upper: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> work_s: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_w: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out_v: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params: EighParams;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  let one_bits = pack_df64_f64(df64_one());
  for (var i = 0u; i < n; i = i + 1u) {
    for (var j = 0u; j < n; j = j + 1u) {
      out_v[i * n + j] = select(vec2<u32>(0u, 0u), one_bits, i == j);
      if (i == j) {
        work_s[i * n + j] = in_a[i * n + j];
      } else if (params.use_upper != 0u) {
        let r = min(i, j);
        let c = max(i, j);
        work_s[i * n + j] = in_a[r * n + c];
      } else {
        let r = max(i, j);
        let c = min(i, j);
        work_s[i * n + j] = in_a[r * n + c];
      }
    }
  }

  let two = df64_from_f32(2.0);
  let one = df64_one();
  for (var sweep = 0u; sweep < 30u; sweep = sweep + 1u) {
    var max_off = 0.0;
    for (var p = 0u; p < n; p = p + 1u) {
      for (var q = p + 1u; q < n; q = q + 1u) {
        let spq = unpack_f64_df64(work_s[p * n + q]);
        let abs_spq = abs(spq.x) + abs(spq.y);
        let spp = unpack_f64_df64(work_s[p * n + p]);
        let sqq = unpack_f64_df64(work_s[q * n + q]);
        let diag_scale = abs(spp.x) + abs(sqq.x);
        if (abs_spq <= 1e-15 * diag_scale) {
          work_s[p * n + q] = vec2<u32>(0u, 0u);
          work_s[q * n + p] = vec2<u32>(0u, 0u);
        } else if (abs_spq > 0.0) {
          if (abs_spq > max_off) {
            max_off = abs_spq;
          }
          let diff = df64_sub(sqq, spp);
          let tau = df64_div(diff, df64_mul(two, spq));
          let abs_tau = df64_abs(tau);
          var t_mag = df64_zero();
          if (abs_tau.x > 1e8) {
            t_mag = df64_div(one, df64_mul(two, abs_tau));
          } else {
            let root = df64_sqrt(df64_add(one, df64_mul(tau, tau)));
            t_mag = df64_div(one, df64_add(abs_tau, root));
          }
          let t = select(df64_neg(t_mag), t_mag, tau.x >= 0.0);
          let c = df64_div(one, df64_sqrt(df64_add(one, df64_mul(t, t))));
          let s = df64_mul(t, c);

          work_s[p * n + p] = pack_df64_f64(df64_sub(spp, df64_mul(t, spq)));
          work_s[q * n + q] = pack_df64_f64(df64_add(sqq, df64_mul(t, spq)));
          work_s[p * n + q] = vec2<u32>(0u, 0u);
          work_s[q * n + p] = vec2<u32>(0u, 0u);

          for (var r = 0u; r < n; r = r + 1u) {
            if (r != p && r != q) {
              let srp = unpack_f64_df64(work_s[r * n + p]);
              let srq = unpack_f64_df64(work_s[r * n + q]);
              let new_rp = pack_df64_f64(df64_sub(df64_mul(c, srp), df64_mul(s, srq)));
              let new_rq = pack_df64_f64(df64_add(df64_mul(s, srp), df64_mul(c, srq)));
              work_s[r * n + p] = new_rp;
              work_s[p * n + r] = new_rp;
              work_s[r * n + q] = new_rq;
              work_s[q * n + r] = new_rq;
            }
          }

          for (var r = 0u; r < n; r = r + 1u) {
            let vrp = unpack_f64_df64(out_v[r * n + p]);
            let vrq = unpack_f64_df64(out_v[r * n + q]);
            out_v[r * n + p] = pack_df64_f64(df64_sub(df64_mul(c, vrp), df64_mul(s, vrq)));
            out_v[r * n + q] = pack_df64_f64(df64_add(df64_mul(s, vrp), df64_mul(c, vrq)));
          }
        }
      }
    }
    if (max_off == 0.0) {
      break;
    }
  }

  for (var i = 0u; i < n; i = i + 1u) {
    out_w[i] = work_s[i * n + i];
  }

  // Sort eigenvalues in ascending order and permute eigenvector columns.
  for (var i = 0u; i < n; i = i + 1u) {
    var min_idx = i;
    var min_val = unpack_f64_df64(out_w[i]);
    for (var j = i + 1u; j < n; j = j + 1u) {
      let wj = unpack_f64_df64(out_w[j]);
      if (df64_lt(wj, min_val)) {
        min_val = wj;
        min_idx = j;
      }
    }
    if (min_idx != i) {
      let tmp_w = out_w[i];
      out_w[i] = out_w[min_idx];
      out_w[min_idx] = tmp_w;
      for (var r = 0u; r < n; r = r + 1u) {
        let tmp_v = out_v[r * n + i];
        out_v[r * n + i] = out_v[r * n + min_idx];
        out_v[r * n + min_idx] = tmp_v;
      }
    }
  }
}
''';

const String _choleskyF32Shader = '''
struct CholParams {
  n: u32,
  upper: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_factor: array<f32>;
@group(0) @binding(2) var<uniform> params: CholParams;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  for (var idx = 0u; idx < n * n; idx = idx + 1u) {
    out_factor[idx] = 0.0;
  }
  for (var i = 0u; i < n; i = i + 1u) {
    for (var j = 0u; j <= i; j = j + 1u) {
      var sum = in_a[i * n + j];
      for (var k = 0u; k < j; k = k + 1u) {
        sum = sum - out_factor[i * n + k] * out_factor[j * n + k];
      }
      if (i == j) {
        out_factor[i * n + i] = sqrt(max(0.0, sum));
      } else {
        out_factor[i * n + j] = sum / out_factor[j * n + j];
      }
    }
  }
  if (params.upper != 0u) {
    for (var i = 0u; i < n; i = i + 1u) {
      for (var j = i + 1u; j < n; j = j + 1u) {
        out_factor[i * n + j] = out_factor[j * n + i];
        out_factor[j * n + i] = 0.0;
      }
    }
  }
}
''';

const String _qrF32Shader = '''
struct QrParams {
  m: u32,
  n: u32,
  q_cols: u32,
  r_rows: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> work_q: array<f32>;
@group(0) @binding(2) var<storage, read_write> work_r: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_q: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_r: array<f32>;
@group(0) @binding(5) var<uniform> params: QrParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let k_min = min(m, n);

  for (var i = 0u; i < m * n; i = i + 1u) {
    work_r[i] = in_a[i];
  }
  for (var r = 0u; r < m; r = r + 1u) {
    for (var c = 0u; c < m; c = c + 1u) {
      work_q[r * m + c] = select(0.0, 1.0, r == c);
    }
  }

  for (var k = 0u; k < k_min; k = k + 1u) {
    var norm_sq = 0.0;
    for (var i = k; i < m; i = i + 1u) {
      let val = work_r[i * n + k];
      norm_sq = norm_sq + val * val;
    }
    let norm_x = sqrt(norm_sq);
    if (norm_x > 0.0) {
      let x0 = work_r[k * n + k];
      let alpha = select(norm_x, -norm_x, x0 >= 0.0);
      let v0 = x0 - alpha;
      var tail_sq = 0.0;
      for (var i = k + 1u; i < m; i = i + 1u) {
        let vi = work_r[i * n + k];
        tail_sq = tail_sq + vi * vi;
      }
      let v_norm_sq = v0 * v0 + tail_sq;
      if (v_norm_sq > 0.0) {
        work_r[k * n + k] = v0;
        for (var c = k + 1u; c < n; c = c + 1u) {
          var dot_rc = 0.0;
          for (var i = k; i < m; i = i + 1u) {
            dot_rc = dot_rc + work_r[i * n + k] * work_r[i * n + c];
          }
          let scale = (2.0 * dot_rc) / v_norm_sq;
          for (var i = k; i < m; i = i + 1u) {
            work_r[i * n + c] = work_r[i * n + c] - scale * work_r[i * n + k];
          }
        }
        for (var r = 0u; r < m; r = r + 1u) {
          var dot_qr = 0.0;
          for (var i = k; i < m; i = i + 1u) {
            dot_qr = dot_qr + work_q[r * m + i] * work_r[i * n + k];
          }
          let scale_q = (2.0 * dot_qr) / v_norm_sq;
          for (var i = k; i < m; i = i + 1u) {
            work_q[r * m + i] = work_q[r * m + i] - scale_q * work_r[i * n + k];
          }
        }
        work_r[k * n + k] = alpha;
        for (var i = k + 1u; i < m; i = i + 1u) {
          work_r[i * n + k] = 0.0;
        }
      }
    }
  }

  let q_cols = params.q_cols;
  for (var r = 0u; r < m; r = r + 1u) {
    for (var c = 0u; c < q_cols; c = c + 1u) {
      out_q[r * q_cols + c] = work_q[r * m + c];
    }
  }
  let r_rows = params.r_rows;
  for (var r = 0u; r < r_rows; r = r + 1u) {
    for (var c = 0u; c < n; c = c + 1u) {
      out_r[r * n + c] = work_r[r * n + c];
    }
  }
}
''';

const String _luDecomposeF32Shader = '''
struct LuParams {
  m: u32,
  n: u32,
  k_min: u32,
  pad: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> work_lu: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_p: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_l: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_u: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_pivots: array<f32>;
@group(0) @binding(6) var<uniform> params: LuParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let k_min = params.k_min;

  for (var i = 0u; i < m * n; i = i + 1u) {
    work_lu[i] = in_a[i];
  }
  for (var r = 0u; r < m; r = r + 1u) {
    for (var c = 0u; c < m; c = c + 1u) {
      out_p[r * m + c] = select(0.0, 1.0, r == c);
    }
  }

  for (var k = 0u; k < k_min; k = k + 1u) {
    var pivot_row = k;
    var max_val = abs(work_lu[k * n + k]);
    for (var r = k + 1u; r < m; r = r + 1u) {
      let cand = abs(work_lu[r * n + k]);
      if (cand > max_val) {
        max_val = cand;
        pivot_row = r;
      }
    }
    out_pivots[k] = f32(pivot_row);

    if (pivot_row != k) {
      for (var c = 0u; c < n; c = c + 1u) {
        let tmp_lu = work_lu[k * n + c];
        work_lu[k * n + c] = work_lu[pivot_row * n + c];
        work_lu[pivot_row * n + c] = tmp_lu;
      }
      for (var r = 0u; r < m; r = r + 1u) {
        let tmp_p = out_p[r * m + k];
        out_p[r * m + k] = out_p[r * m + pivot_row];
        out_p[r * m + pivot_row] = tmp_p;
      }
    }

    let piv = work_lu[k * n + k];
    if (piv != 0.0) {
      for (var i = k + 1u; i < m; i = i + 1u) {
        let mult = work_lu[i * n + k] / piv;
        work_lu[i * n + k] = mult;
        for (var j = k + 1u; j < n; j = j + 1u) {
          work_lu[i * n + j] = work_lu[i * n + j] - mult * work_lu[k * n + j];
        }
      }
    }
  }

  for (var i = 0u; i < m; i = i + 1u) {
    for (var j = 0u; j < k_min; j = j + 1u) {
      if (i > j) {
        out_l[i * k_min + j] = work_lu[i * n + j];
      } else if (i == j) {
        out_l[i * k_min + j] = 1.0;
      } else {
        out_l[i * k_min + j] = 0.0;
      }
    }
  }
  for (var i = 0u; i < k_min; i = i + 1u) {
    for (var j = 0u; j < n; j = j + 1u) {
      if (j >= i) {
        out_u[i * n + j] = work_lu[i * n + j];
      } else {
        out_u[i * n + j] = 0.0;
      }
    }
  }
}
''';

const String _luSolveF32Shader = '''
struct LuSolveParams {
  n: u32,
  nrhs: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_lu: array<f32>;
@group(0) @binding(1) var<storage, read> in_pivots: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_x: array<f32>;
@group(0) @binding(4) var<uniform> params: LuSolveParams;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  let nrhs = params.nrhs;

  for (var i = 0u; i < n * nrhs; i = i + 1u) {
    out_x[i] = in_b[i];
  }

  for (var k = 0u; k < n; k = k + 1u) {
    let piv = u32(max(0.0, round(in_pivots[k])));
    if (piv != k && piv < n) {
      for (var c = 0u; c < nrhs; c = c + 1u) {
        let tmp = out_x[k * nrhs + c];
        out_x[k * nrhs + c] = out_x[piv * nrhs + c];
        out_x[piv * nrhs + c] = tmp;
      }
    }
  }

  for (var c = 0u; c < nrhs; c = c + 1u) {
    for (var i = 0u; i < n; i = i + 1u) {
      var sum = out_x[i * nrhs + c];
      for (var j = 0u; j < i; j = j + 1u) {
        sum = sum - in_lu[i * n + j] * out_x[j * nrhs + c];
      }
      out_x[i * nrhs + c] = sum;
    }

    if (n > 0u) {
      var i = i32(n) - 1;
      loop {
        if (i < 0) { break; }
        let ui = u32(i);
        var sum = out_x[ui * nrhs + c];
        for (var j = ui + 1u; j < n; j = j + 1u) {
          sum = sum - in_lu[ui * n + j] * out_x[j * nrhs + c];
        }
        out_x[ui * nrhs + c] = sum / in_lu[ui * n + ui];
        i = i - 1;
      }
    }
  }
}
''';

const String _eighF32Shader = '''
struct EighParams {
  n: u32,
  use_upper: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> work_s: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_v: array<f32>;
@group(0) @binding(4) var<uniform> params: EighParams;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  for (var i = 0u; i < n; i = i + 1u) {
    for (var j = 0u; j < n; j = j + 1u) {
      out_v[i * n + j] = select(0.0, 1.0, i == j);
      if (i == j) {
        work_s[i * n + j] = in_a[i * n + j];
      } else if (params.use_upper != 0u) {
        let r = min(i, j);
        let c = max(i, j);
        work_s[i * n + j] = in_a[r * n + c];
      } else {
        let r = max(i, j);
        let c = min(i, j);
        work_s[i * n + j] = in_a[r * n + c];
      }
    }
  }

  for (var sweep = 0u; sweep < 30u; sweep = sweep + 1u) {
    var max_off = 0.0;
    for (var p = 0u; p < n; p = p + 1u) {
      for (var q = p + 1u; q < n; q = q + 1u) {
        let spq = work_s[p * n + q];
        let abs_spq = abs(spq);
        let spp = work_s[p * n + p];
        let sqq = work_s[q * n + q];
        let diag_scale = abs(spp) + abs(sqq);
        if (abs_spq <= 1e-7 * diag_scale) {
          work_s[p * n + q] = 0.0;
          work_s[q * n + p] = 0.0;
        } else if (abs_spq > 0.0) {
          if (abs_spq > max_off) {
            max_off = abs_spq;
          }
          let diff = sqq - spp;
          let tau = diff / (2.0 * spq);
          let abs_tau = abs(tau);
          var t_mag = 0.0;
          if (abs_tau > 1e4) {
            t_mag = 1.0 / (2.0 * abs_tau);
          } else {
            t_mag = 1.0 / (abs_tau + sqrt(1.0 + tau * tau));
          }
          let t = select(-t_mag, t_mag, tau >= 0.0);
          let c = 1.0 / sqrt(1.0 + t * t);
          let s = t * c;

          work_s[p * n + p] = spp - t * spq;
          work_s[q * n + q] = sqq + t * spq;
          work_s[p * n + q] = 0.0;
          work_s[q * n + p] = 0.0;

          for (var r = 0u; r < n; r = r + 1u) {
            if (r != p && r != q) {
              let srp = work_s[r * n + p];
              let srq = work_s[r * n + q];
              let new_rp = c * srp - s * srq;
              let new_rq = s * srp + c * srq;
              work_s[r * n + p] = new_rp;
              work_s[p * n + r] = new_rp;
              work_s[r * n + q] = new_rq;
              work_s[q * n + r] = new_rq;
            }
          }

          for (var r = 0u; r < n; r = r + 1u) {
            let vrp = out_v[r * n + p];
            let vrq = out_v[r * n + q];
            out_v[r * n + p] = c * vrp - s * vrq;
            out_v[r * n + q] = s * vrp + c * vrq;
          }
        }
      }
    }
    if (max_off == 0.0) {
      break;
    }
  }

  for (var i = 0u; i < n; i = i + 1u) {
    out_w[i] = work_s[i * n + i];
  }

  for (var i = 0u; i < n; i = i + 1u) {
    var min_idx = i;
    var min_val = out_w[i];
    for (var j = i + 1u; j < n; j = j + 1u) {
      let wj = out_w[j];
      if (wj < min_val) {
        min_val = wj;
        min_idx = j;
      }
    }
    if (min_idx != i) {
      let tmp_w = out_w[i];
      out_w[i] = out_w[min_idx];
      out_w[min_idx] = tmp_w;
      for (var r = 0u; r < n; r = r + 1u) {
        let tmp_v = out_v[r * n + i];
        out_v[r * n + i] = out_v[r * n + min_idx];
        out_v[r * n + min_idx] = tmp_v;
      }
    }
  }
}
''';

/// Dispatches the Cholesky decomposition kernel on [device].
GpuBuffer dispatchCholeskyGpu(
  GpuDevice device,
  GpuBuffer inputBuffer,
  int n, {
  required bool upper,
  bool singlePrecision = false,
}) {
  final elemBytes = singlePrecision ? 4 : 8;
  final outputBuffer = device.createBuffer(
    sizeInBytes: math.max(1, n * n) * elemBytes,
  );
  if (n == 0) return outputBuffer;

  final module = singlePrecision
      ? getOrCreateLinalgShader('linalg_cholesky_f32', () => _choleskyF32Shader)
      : getOrCreateLinalgShader(
          'linalg_cholesky_f64',
          () => _choleskyRealShader,
        );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputBuffer, outputBuffer],
    uniforms: [n, upper ? 1 : 0, 0, 0],
    workgroupsX: 1,
  );
  return outputBuffer;
}

/// Dispatches the Householder QR decomposition kernel on [device].
({GpuBuffer q, GpuBuffer r}) dispatchQrGpu(
  GpuDevice device,
  GpuBuffer inputBuffer,
  int m,
  int n, {
  required int qCols,
  required int rRows,
  bool singlePrecision = false,
}) {
  final elemBytes = singlePrecision ? 4 : 8;
  final workQ = device.createBuffer(
    sizeInBytes: math.max(1, m * m) * elemBytes,
  );
  final workR = device.createBuffer(
    sizeInBytes: math.max(1, m * n) * elemBytes,
  );
  final outQ = device.createBuffer(
    sizeInBytes: math.max(1, m * qCols) * elemBytes,
  );
  final outR = device.createBuffer(
    sizeInBytes: math.max(1, rRows * n) * elemBytes,
  );
  if (m == 0 || n == 0) return (q: outQ, r: outR);

  final module = singlePrecision
      ? getOrCreateLinalgShader('linalg_qr_f32', () => _qrF32Shader)
      : getOrCreateLinalgShader('linalg_qr_f64', () => _qrRealShader);
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputBuffer, workQ, workR, outQ, outR],
    uniforms: [m, n, qCols, rRows],
    workgroupsX: 1,
  );
  return (q: outQ, r: outR);
}

/// Dispatches the LU decomposition kernel on [device].
({GpuBuffer lu, GpuBuffer p, GpuBuffer l, GpuBuffer u, GpuBuffer pivots})
dispatchLuGpu(
  GpuDevice device,
  GpuBuffer inputBuffer,
  int m,
  int n, {
  bool singlePrecision = false,
}) {
  final elemBytes = singlePrecision ? 4 : 8;
  final kMin = math.min(m, n);
  final workLu = device.createBuffer(
    sizeInBytes: math.max(1, m * n) * elemBytes,
  );
  final outP = device.createBuffer(sizeInBytes: math.max(1, m * m) * elemBytes);
  final outL = device.createBuffer(
    sizeInBytes: math.max(1, m * kMin) * elemBytes,
  );
  final outU = device.createBuffer(
    sizeInBytes: math.max(1, kMin * n) * elemBytes,
  );
  final outPivots = device.createBuffer(
    sizeInBytes: math.max(1, kMin) * elemBytes,
  );
  if (m == 0 || n == 0) {
    return (lu: workLu, p: outP, l: outL, u: outU, pivots: outPivots);
  }

  final module = singlePrecision
      ? getOrCreateLinalgShader(
          'linalg_lu_decompose_f32',
          () => _luDecomposeF32Shader,
        )
      : getOrCreateLinalgShader(
          'linalg_lu_decompose_f64',
          () => _luDecomposeShader,
        );
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputBuffer, workLu, outP, outL, outU, outPivots],
    uniforms: [m, n, kMin, 0],
    workgroupsX: 1,
  );
  return (lu: workLu, p: outP, l: outL, u: outU, pivots: outPivots);
}

/// Dispatches the LU linear system solver kernel on [device].
GpuBuffer dispatchLuSolveGpu(
  GpuDevice device,
  GpuBuffer luBuffer,
  GpuBuffer pivotsBuffer,
  GpuBuffer bBuffer,
  int n,
  int nrhs, {
  bool singlePrecision = false,
}) {
  final elemBytes = singlePrecision ? 4 : 8;
  final outX = device.createBuffer(
    sizeInBytes: math.max(1, n * nrhs) * elemBytes,
  );
  if (n == 0 || nrhs == 0) return outX;

  final module = singlePrecision
      ? getOrCreateLinalgShader('linalg_lu_solve_f32', () => _luSolveF32Shader)
      : getOrCreateLinalgShader('linalg_lu_solve_f64', () => _luSolveShader);
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [luBuffer, pivotsBuffer, bBuffer, outX],
    uniforms: [n, nrhs, 0, 0],
    workgroupsX: 1,
  );
  return outX;
}

/// Dispatches the symmetric Jacobi eigendecomposition kernel on [device].
({GpuBuffer eigenvalues, GpuBuffer eigenvectors}) dispatchEighGpu(
  GpuDevice device,
  GpuBuffer inputBuffer,
  int n, {
  required bool useUpper,
  bool singlePrecision = false,
}) {
  final elemBytes = singlePrecision ? 4 : 8;
  final workS = device.createBuffer(
    sizeInBytes: math.max(1, n * n) * elemBytes,
  );
  final outW = device.createBuffer(sizeInBytes: math.max(1, n) * elemBytes);
  final outV = device.createBuffer(sizeInBytes: math.max(1, n * n) * elemBytes);
  if (n == 0) return (eigenvalues: outW, eigenvectors: outV);

  final module = singlePrecision
      ? getOrCreateLinalgShader('linalg_eigh_f32', () => _eighF32Shader)
      : getOrCreateLinalgShader('linalg_eigh_f64', () => _eighRealShader);
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputBuffer, workS, outW, outV],
    uniforms: [n, useUpper ? 1 : 0, 0, 0],
    workgroupsX: 1,
  );
  return (eigenvalues: outW, eigenvectors: outV);
}
