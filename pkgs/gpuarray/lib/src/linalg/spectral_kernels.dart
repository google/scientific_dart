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

const String _svdRealShader =
    '''
$linalgDf64WgslLibrary

struct SvdParams {
  m: u32,
  n: u32,
  u_cols: u32,
  vt_rows: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> work_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> work_u_full: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> work_v_full: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out_u: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> out_s: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> out_vt: array<vec2<u32>>;
@group(0) @binding(7) var<uniform> params: SvdParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let transposed = m < n;
  let p = max(m, n);
  let q = min(m, n);
  let one = df64_one();
  let two = df64_from_f32(2.0);
  let one_bits = pack_df64_f64(one);

  // Load M (p x q) into work_b: M = A if m >= n, else A^T.
  for (var r = 0u; r < p; r = r + 1u) {
    for (var c = 0u; c < q; c = c + 1u) {
      if (!transposed) {
        work_b[r * q + c] = in_a[r * n + c];
      } else {
        work_b[r * q + c] = in_a[c * n + r];
      }
    }
  }

  // Initialize work_v_full (q x q) to identity.
  for (var r = 0u; r < q; r = r + 1u) {
    for (var c = 0u; c < q; c = c + 1u) {
      work_v_full[r * q + c] = select(vec2<u32>(0u, 0u), one_bits, r == c);
    }
  }

  // One-sided Hestenes Jacobi rotations on columns of work_b (p x q).
  for (var sweep = 0u; sweep < 35u; sweep = sweep + 1u) {
    var max_off = 0.0;
    for (var i = 0u; i < q; i = i + 1u) {
      for (var j = i + 1u; j < q; j = j + 1u) {
        var app = df64_zero();
        var aqq = df64_zero();
        var apq = df64_zero();
        for (var r = 0u; r < p; r = r + 1u) {
          let bri = unpack_f64_df64(work_b[r * q + i]);
          let brj = unpack_f64_df64(work_b[r * q + j]);
          app = df64_add(app, df64_mul(bri, bri));
          aqq = df64_add(aqq, df64_mul(brj, brj));
          apq = df64_add(apq, df64_mul(bri, brj));
        }
        let abs_apq = abs(apq.x) + abs(apq.y);
        let scale_ij = sqrt(max(0.0, app.x * aqq.x));
        if (abs_apq > 1e-15 * scale_ij && abs_apq > 0.0) {
          if (abs_apq > max_off) {
            max_off = abs_apq;
          }
          let diff = df64_sub(aqq, app);
          let tau = df64_div(diff, df64_mul(two, apq));
          let abs_tau = df64_abs(tau);
          var t_mag = df64_zero();
          if (abs_tau.x > 1e8) {
            t_mag = df64_div(one, df64_mul(two, abs_tau));
          } else {
            let root = df64_sqrt(df64_add(one, df64_mul(tau, tau)));
            t_mag = df64_div(one, df64_add(abs_tau, root));
          }
          let t = select(df64_neg(t_mag), t_mag, tau.x >= 0.0);
          let c_rot = df64_div(one, df64_sqrt(df64_add(one, df64_mul(t, t))));
          let s_rot = df64_mul(t, c_rot);

          for (var r = 0u; r < p; r = r + 1u) {
            let bri = unpack_f64_df64(work_b[r * q + i]);
            let brj = unpack_f64_df64(work_b[r * q + j]);
            work_b[r * q + i] = pack_df64_f64(df64_sub(df64_mul(c_rot, bri), df64_mul(s_rot, brj)));
            work_b[r * q + j] = pack_df64_f64(df64_add(df64_mul(s_rot, bri), df64_mul(c_rot, brj)));
          }
          for (var r = 0u; r < q; r = r + 1u) {
            let vri = unpack_f64_df64(work_v_full[r * q + i]);
            let vrj = unpack_f64_df64(work_v_full[r * q + j]);
            work_v_full[r * q + i] = pack_df64_f64(df64_sub(df64_mul(c_rot, vri), df64_mul(s_rot, vrj)));
            work_v_full[r * q + j] = pack_df64_f64(df64_add(df64_mul(s_rot, vri), df64_mul(c_rot, vrj)));
          }
        }
      }
    }
    if (max_off == 0.0) {
      break;
    }
  }

  // Compute singular values from column norms of work_b.
  for (var j = 0u; j < q; j = j + 1u) {
    var sum_sq = df64_zero();
    for (var r = 0u; r < p; r = r + 1u) {
      let brj = unpack_f64_df64(work_b[r * q + j]);
      sum_sq = df64_add(sum_sq, df64_mul(brj, brj));
    }
    out_s[j] = pack_df64_f64(df64_sqrt(sum_sq));
  }

  // Sort singular values in descending order and permute columns of work_b and work_v_full.
  for (var i = 0u; i < q; i = i + 1u) {
    var max_idx = i;
    var max_val = unpack_f64_df64(out_s[i]);
    for (var j = i + 1u; j < q; j = j + 1u) {
      let sj = unpack_f64_df64(out_s[j]);
      if (df64_gt(sj, max_val)) {
        max_val = sj;
        max_idx = j;
      }
    }
    if (max_idx != i) {
      let tmp_s = out_s[i];
      out_s[i] = out_s[max_idx];
      out_s[max_idx] = tmp_s;
      for (var r = 0u; r < p; r = r + 1u) {
        let tmp_b = work_b[r * q + i];
        work_b[r * q + i] = work_b[r * q + max_idx];
        work_b[r * q + max_idx] = tmp_b;
      }
      for (var r = 0u; r < q; r = r + 1u) {
        let tmp_v = work_v_full[r * q + i];
        work_v_full[r * q + i] = work_v_full[r * q + max_idx];
        work_v_full[r * q + max_idx] = tmp_v;
      }
    }
  }

  // Build orthonormal work_u_full (p x p).
  let s0 = unpack_f64_df64(out_s[0]);
  let tol = max(1e-12 * s0.x, 1e-30);
  var rank_count = 0u;
  for (var j = 0u; j < q; j = j + 1u) {
    let sj = unpack_f64_df64(out_s[j]);
    if (sj.x > tol) {
      for (var r = 0u; r < p; r = r + 1u) {
        let brj = unpack_f64_df64(work_b[r * q + j]);
        work_u_full[r * p + j] = pack_df64_f64(df64_div(brj, sj));
      }
      rank_count = j + 1u;
    } else {
      break;
    }
  }

  // Complete remaining columns rank_count .. p - 1 of work_u_full via Gram-Schmidt.
  for (var col = rank_count; col < p; col = col + 1u) {
    var best_norm = -1.0;
    for (var cand = 0u; cand < p; cand = cand + 1u) {
      // Try standard basis vector e_cand in column p-1 temporarily or directly in col.
      for (var r = 0u; r < p; r = r + 1u) {
        work_u_full[r * p + col] = select(vec2<u32>(0u, 0u), one_bits, r == cand);
      }
      for (var gs_pass = 0u; gs_pass < 2u; gs_pass = gs_pass + 1u) {
        for (var prev = 0u; prev < col; prev = prev + 1u) {
          var dot_val = df64_zero();
          for (var r = 0u; r < p; r = r + 1u) {
            let ur_prev = unpack_f64_df64(work_u_full[r * p + prev]);
            let ur_col = unpack_f64_df64(work_u_full[r * p + col]);
            dot_val = df64_add(dot_val, df64_mul(ur_prev, ur_col));
          }
          for (var r = 0u; r < p; r = r + 1u) {
            let ur_prev = unpack_f64_df64(work_u_full[r * p + prev]);
            let ur_col = unpack_f64_df64(work_u_full[r * p + col]);
            work_u_full[r * p + col] = pack_df64_f64(df64_sub(ur_col, df64_mul(dot_val, ur_prev)));
          }
        }
      }
      var norm_sq = df64_zero();
      for (var r = 0u; r < p; r = r + 1u) {
        let ur_col = unpack_f64_df64(work_u_full[r * p + col]);
        norm_sq = df64_add(norm_sq, df64_mul(ur_col, ur_col));
      }
      if (norm_sq.x > 0.1) {
        let inv_norm = df64_div(one, df64_sqrt(norm_sq));
        for (var r = 0u; r < p; r = r + 1u) {
          let ur_col = unpack_f64_df64(work_u_full[r * p + col]);
          work_u_full[r * p + col] = pack_df64_f64(df64_mul(ur_col, inv_norm));
        }
        best_norm = norm_sq.x;
        break;
      }
    }
  }

  let u_cols = params.u_cols;
  let vt_rows = params.vt_rows;
  if (!transposed) {
    // m >= n: U is m x u_cols from work_u_full, Vt is vt_rows x n from work_v_full^T.
    for (var r = 0u; r < m; r = r + 1u) {
      for (var c = 0u; c < u_cols; c = c + 1u) {
        out_u[r * u_cols + c] = work_u_full[r * p + c];
      }
    }
    for (var r = 0u; r < vt_rows; r = r + 1u) {
      for (var c = 0u; c < n; c = c + 1u) {
        out_vt[r * n + c] = work_v_full[c * q + r];
      }
    }
  } else {
    // m < n: M = A^T = U_M * S * V_M^T => A = V_M * S * U_M^T.
    for (var r = 0u; r < m; r = r + 1u) {
      for (var c = 0u; c < u_cols; c = c + 1u) {
        out_u[r * u_cols + c] = work_v_full[r * q + c];
      }
    }
    for (var r = 0u; r < vt_rows; r = r + 1u) {
      for (var c = 0u; c < n; c = c + 1u) {
        out_vt[r * n + c] = work_u_full[c * p + r];
      }
    }
  }
}
''';

const String _eigRealShader =
    '''
$linalgDf64WgslLibrary

struct EigParams {
  n: u32,
  compute_vectors: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> work_h: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> work_q: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> work_m: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out_w: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> out_v: array<vec2<u32>>;
@group(0) @binding(6) var<uniform> params: EigParams;

fn get_h(n: u32, r: u32, c: u32) -> vec2<f32> {
  return unpack_f64_df64(work_h[r * n + c]);
}

fn set_h(n: u32, r: u32, c: u32, val: vec2<f32>) {
  work_h[r * n + c] = pack_df64_f64(val);
}

fn get_q(n: u32, r: u32, c: u32) -> vec2<f32> {
  return unpack_f64_df64(work_q[r * n + c]);
}

fn set_q(n: u32, r: u32, c: u32, val: vec2<f32>) {
  work_q[r * n + c] = pack_df64_f64(val);
}

fn get_m_c(n: u32, r: u32, c: u32) -> vec4<f32> {
  let base = 2u * (r * n + c);
  let re = unpack_f64_df64(work_m[base]);
  let im = unpack_f64_df64(work_m[base + 1u]);
  return vec4<f32>(re.x, re.y, im.x, im.y);
}

fn set_m_c(n: u32, r: u32, c: u32, val: vec4<f32>) {
  let base = 2u * (r * n + c);
  work_m[base] = pack_df64_f64(vec2<f32>(val.x, val.y));
  work_m[base + 1u] = pack_df64_f64(vec2<f32>(val.z, val.w));
}

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  let one = df64_one();
  let two = df64_from_f32(2.0);
  let half = df64_from_f32(0.5);
  let one_bits = pack_df64_f64(one);

  for (var r = 0u; r < n; r = r + 1u) {
    for (var c = 0u; c < n; c = c + 1u) {
      work_h[r * n + c] = in_a[r * n + c];
      work_q[r * n + c] = select(vec2<u32>(0u, 0u), one_bits, r == c);
    }
  }

  // Step 1: Upper Hessenberg reduction H = Q^T * A * Q.
  if (n > 2u) {
    for (var k = 0u; k < n - 2u; k = k + 1u) {
      var norm_sq = df64_zero();
      for (var i = k + 1u; i < n; i = i + 1u) {
        let hik = get_h(n, i, k);
        norm_sq = df64_add(norm_sq, df64_mul(hik, hik));
      }
      let norm_x = df64_sqrt(norm_sq);
      if (norm_x.x > 0.0 || norm_x.y > 0.0) {
        let x0 = get_h(n, k + 1u, k);
        let alpha = select(norm_x, df64_neg(norm_x), x0.x >= 0.0);
        let v0 = df64_sub(x0, alpha);
        var tail_sq = df64_zero();
        for (var i = k + 2u; i < n; i = i + 1u) {
          let vi = get_h(n, i, k);
          tail_sq = df64_add(tail_sq, df64_mul(vi, vi));
        }
        let v_norm_sq = df64_add(df64_mul(v0, v0), tail_sq);
        if (v_norm_sq.x > 0.0 || v_norm_sq.y > 0.0) {
          set_h(n, k + 1u, k, v0);
          // Left reflection: H[k+1..n, k+1..n]
          for (var c = k + 1u; c < n; c = c + 1u) {
            var dot_v = df64_zero();
            for (var i = k + 1u; i < n; i = i + 1u) {
              dot_v = df64_add(dot_v, df64_mul(get_h(n, i, k), get_h(n, i, c)));
            }
            let scale = df64_div(df64_mul(two, dot_v), v_norm_sq);
            for (var i = k + 1u; i < n; i = i + 1u) {
              set_h(n, i, c, df64_sub(get_h(n, i, c), df64_mul(scale, get_h(n, i, k))));
            }
          }
          // Right reflection: H[0..n, k+1..n]
          for (var r = 0u; r < n; r = r + 1u) {
            var dot_v = df64_zero();
            for (var i = k + 1u; i < n; i = i + 1u) {
              dot_v = df64_add(dot_v, df64_mul(get_h(n, r, i), get_h(n, i, k)));
            }
            let scale = df64_div(df64_mul(two, dot_v), v_norm_sq);
            for (var i = k + 1u; i < n; i = i + 1u) {
              set_h(n, r, i, df64_sub(get_h(n, r, i), df64_mul(scale, get_h(n, i, k))));
            }
          }
          // Accumulate Q[0..n, k+1..n]
          for (var r = 0u; r < n; r = r + 1u) {
            var dot_v = df64_zero();
            for (var i = k + 1u; i < n; i = i + 1u) {
              dot_v = df64_add(dot_v, df64_mul(get_q(n, r, i), get_h(n, i, k)));
            }
            let scale = df64_div(df64_mul(two, dot_v), v_norm_sq);
            for (var i = k + 1u; i < n; i = i + 1u) {
              set_q(n, r, i, df64_sub(get_q(n, r, i), df64_mul(scale, get_h(n, i, k))));
            }
          }
          set_h(n, k + 1u, k, alpha);
          for (var i = k + 2u; i < n; i = i + 1u) {
            work_h[i * n + k] = vec2<u32>(0u, 0u);
          }
        }
      }
    }
  }

  // Step 2: Shifted QR iteration to real Schur form.
  if (n > 1u) {
    var active_end = i32(n) - 1;
    var iter = 0u;
    loop {
      if (active_end <= 0 || iter >= 300u) {
        break;
      }
      // Deflate negligible subdiagonals.
      var l = active_end;
      loop {
        if (l <= 0) { break; }
        let sub = abs(get_h(n, u32(l), u32(l - 1)).x);
        let diag_sum = abs(get_h(n, u32(l - 1), u32(l - 1)).x) + abs(get_h(n, u32(l), u32(l)).x);
        if (sub <= 1e-13 * max(1e-20, diag_sum)) {
          work_h[u32(l) * n + u32(l - 1)] = vec2<u32>(0u, 0u);
          break;
        }
        l = l - 1;
      }
      if (l == active_end) {
        active_end = active_end - 1;
        continue;
      }
      if (l == active_end - 1) {
        // 2x2 block at [active_end - 1, active_end]: check if real or complex pair.
        let p_idx = u32(active_end - 1);
        let q_idx = u32(active_end);
        let a = get_h(n, p_idx, p_idx);
        let b = get_h(n, p_idx, q_idx);
        let c = get_h(n, q_idx, p_idx);
        let d = get_h(n, q_idx, q_idx);
        let half_diff = df64_mul(half, df64_sub(a, d));
        let disc = df64_add(df64_mul(half_diff, half_diff), df64_mul(b, c));
        if (disc.x >= 0.0) {
          // Real eigenvalues: apply a single Givens similarity rotation to zero out c.
          let root = df64_sqrt(disc);
          let shift = df64_add(d, select(df64_add(half_diff, root), df64_sub(half_diff, root), half_diff.x < 0.0));
          // Vector (a - shift, c) or (b, d - shift)
          var vx = df64_sub(a, shift);
          var vy = c;
          if (abs(b.x) > abs(c.x)) {
            vx = b;
            vy = df64_sub( shift, a );
          }
          let r_len = df64_sqrt(df64_add(df64_mul(vx, vx), df64_mul(vy, vy)));
          if (r_len.x > 0.0) {
            let cs = df64_div(vx, r_len);
            let sn = df64_div(vy, r_len);
            for (var col = p_idx; col < n; col = col + 1u) {
              let hp = get_h(n, p_idx, col);
              let hq = get_h(n, q_idx, col);
              set_h(n, p_idx, col, df64_add(df64_mul(cs, hp), df64_mul(sn, hq)));
              set_h(n, q_idx, col, df64_sub(df64_mul(cs, hq), df64_mul(sn, hp)));
            }
            for (var row = 0u; row <= q_idx; row = row + 1u) {
              let hp = get_h(n, row, p_idx);
              let hq = get_h(n, row, q_idx);
              set_h(n, row, p_idx, df64_add(df64_mul(cs, hp), df64_mul(sn, hq)));
              set_h(n, row, q_idx, df64_sub(df64_mul(cs, hq), df64_mul(sn, hp)));
            }
            for (var row = 0u; row < n; row = row + 1u) {
              let qp = get_q(n, row, p_idx);
              let qq = get_q(n, row, q_idx);
              set_q(n, row, p_idx, df64_add(df64_mul(cs, qp), df64_mul(sn, qq)));
              set_q(n, row, q_idx, df64_sub(df64_mul(cs, qq), df64_mul(sn, qp)));
            }
          }
          work_h[q_idx * n + p_idx] = vec2<u32>(0u, 0u);
        }
        active_end = active_end - 2;
        continue;
      }

      // Active block is [l .. active_end] of size >= 3.
      iter = iter + 1u;
      let p_idx = u32(active_end - 1);
      let q_idx = u32(active_end);
      let a = get_h(n, p_idx, p_idx);
      let b = get_h(n, p_idx, q_idx);
      let c = get_h(n, q_idx, p_idx);
      let d = get_h(n, q_idx, q_idx);
      let half_diff = df64_mul(half, df64_sub(a, d));
      let disc = df64_add(df64_mul(half_diff, half_diff), df64_mul(b, c));
      var shift = d;
      if (disc.x >= 0.0) {
        let root = df64_sqrt(disc);
        shift = df64_add(d, select(df64_sub(half_diff, root), df64_add(half_diff, root), half_diff.x >= 0.0));
      } else if ((iter % 10u) == 0u) {
        shift = df64_add(d, df64_abs(c));
      }

      // Single-shift QR sweep on [l .. active_end] using Givens rotations.
      let u_start = u32(l);
      let u_end = u32(active_end);
      for (var i = u_start; i <= u_end; i = i + 1u) {
        set_h(n, i, i, df64_sub(get_h(n, i, i), shift));
      }
      // Save cosines/sines in out_w temporarily (2 * n slots of vec2<u32>).
      for (var i = u_start; i < u_end; i = i + 1u) {
        let x_val = get_h(n, i, i);
        let y_val = get_h(n, i + 1u, i);
        let r_len = df64_sqrt(df64_add(df64_mul(x_val, x_val), df64_mul(y_val, y_val)));
        var cs = one;
        var sn = df64_zero();
        if (r_len.x > 0.0 || r_len.y > 0.0) {
          cs = df64_div(x_val, r_len);
          sn = df64_div(y_val, r_len);
        }
        out_w[2u * i] = pack_df64_f64(cs);
        out_w[2u * i + 1u] = pack_df64_f64(sn);
        for (var col = i; col < n; col = col + 1u) {
          let hi = get_h(n, i, col);
          let hi1 = get_h(n, i + 1u, col);
          set_h(n, i, col, df64_add(df64_mul(cs, hi), df64_mul(sn, hi1)));
          set_h(n, i + 1u, col, df64_sub(df64_mul(cs, hi1), df64_mul(sn, hi)));
        }
        work_h[(i + 1u) * n + i] = vec2<u32>(0u, 0u);
      }
      for (var i = u_start; i < u_end; i = i + 1u) {
        let cs = unpack_f64_df64(out_w[2u * i]);
        let sn = unpack_f64_df64(out_w[2u * i + 1u]);
        for (var row = 0u; row <= i + 1u; row = row + 1u) {
          let hi = get_h(n, row, i);
          let hi1 = get_h(n, row, i + 1u);
          set_h(n, row, i, df64_add(df64_mul(cs, hi), df64_mul(sn, hi1)));
          set_h(n, row, i + 1u, df64_sub(df64_mul(cs, hi1), df64_mul(sn, hi)));
        }
        for (var row = 0u; row < n; row = row + 1u) {
          let qi = get_q(n, row, i);
          let qi1 = get_q(n, row, i + 1u);
          set_q(n, row, i, df64_add(df64_mul(cs, qi), df64_mul(sn, qi1)));
          set_q(n, row, i + 1u, df64_sub(df64_mul(cs, qi1), df64_mul(sn, qi)));
        }
      }
      for (var i = u_start; i <= u_end; i = i + 1u) {
        set_h(n, i, i, df64_add(get_h(n, i, i), shift));
      }
    }
  }

  // Step 3: Extract eigenvalues from real Schur form H.
  var idx = 0u;
  loop {
    if (idx >= n) { break; }
    if (idx + 1u < n) {
      let sub = get_h(n, idx + 1u, idx);
      if (abs(sub.x) > 1e-12) {
        let a = get_h(n, idx, idx);
        let b = get_h(n, idx, idx + 1u);
        let c = sub;
        let d = get_h(n, idx + 1u, idx + 1u);
        let mean = df64_mul(half, df64_add(a, d));
        let half_diff = df64_mul(half, df64_sub(a, d));
        let disc = df64_add(df64_mul(half_diff, half_diff), df64_mul(b, c));
        let imag = df64_sqrt(df64_abs(disc));
        out_w[2u * idx] = pack_df64_f64(mean);
        out_w[2u * idx + 1u] = pack_df64_f64(imag);
        out_w[2u * (idx + 1u)] = pack_df64_f64(mean);
        out_w[2u * (idx + 1u) + 1u] = pack_df64_f64(df64_neg(imag));
        idx = idx + 2u;
        continue;
      }
    }
    out_w[2u * idx] = work_h[idx * n + idx];
    out_w[2u * idx + 1u] = vec2<u32>(0u, 0u);
    idx = idx + 1u;
  }

  if (params.compute_vectors == 0u) {
    return;
  }

  // Step 4: Compute eigenvectors of Schur matrix H for each eigenvalue and transform by Q.
  for (var eig_idx = 0u; eig_idx < n; eig_idx = eig_idx + 1u) {
    let lambda_re = unpack_f64_df64(out_w[2u * eig_idx]);
    let lambda_im = unpack_f64_df64(out_w[2u * eig_idx + 1u]);
    let lambda = vec4<f32>(lambda_re.x, lambda_re.y, lambda_im.x, lambda_im.y);

    var m_end = eig_idx;
    if (eig_idx + 1u < n && abs(get_h(n, eig_idx + 1u, eig_idx).x) > 1e-12) {
      m_end = eig_idx + 1u;
    }
    let m_sz = m_end + 1u;

    for (var r = 0u; r < m_sz; r = r + 1u) {
      for (var c = 0u; c < m_sz; c = c + 1u) {
        var val = cdf64_from_df64(get_h(n, r, c), df64_zero());
        if (r == c) {
          val = cdf64_sub(val, lambda);
        }
        set_m_c(n, r, c, val);
      }
    }

    // Eliminate subdiagonal of M[0..m_sz, 0..m_sz] with partial pivoting between r and r+1.
    if (m_sz > 1u) {
      for (var k = 0u; k < m_sz - 1u; k = k + 1u) {
        let diag_k = get_m_c(n, k, k);
        let sub_k = get_m_c(n, k + 1u, k);
        let abs_diag = cdf64_abs2(diag_k).x;
        let abs_sub = cdf64_abs2(sub_k).x;
        if (abs_sub > abs_diag) {
          for (var c = k; c < m_sz; c = c + 1u) {
            let tmp = get_m_c(n, k, c);
            set_m_c(n, k, c, get_m_c(n, k + 1u, c));
            set_m_c(n, k + 1u, c, tmp);
          }
        }
        let piv = get_m_c(n, k, k);
        let elim = get_m_c(n, k + 1u, k);
        if (cdf64_abs2(piv).x > 1e-28 && cdf64_abs2(elim).x > 0.0) {
          let factor = cdf64_div(elim, piv);
          set_m_c(n, k + 1u, k, cdf64_zero());
          for (var c = k + 1u; c < m_sz; c = c + 1u) {
            let updated = cdf64_sub(get_m_c(n, k + 1u, c), cdf64_mul(factor, get_m_c(n, k, c)));
            set_m_c(n, k + 1u, c, updated);
          }
        }
      }
    }

    // Back-substitution in row n-1 of work_m (using last row as temporary complex vector u[0..n]).
    // Wait: if m_sz == n, row n-1 of work_m is used only for r = m_sz - 1 which has u[m_sz - 1] = 1!
    // Store u[c] in out_v[c * n + eig_idx] temporarily before multiplying by Q.
    for (var i = 0u; i < n; i = i + 1u) {
      out_v[2u * (i * n + eig_idx)] = vec2<u32>(0u, 0u);
      out_v[2u * (i * n + eig_idx) + 1u] = vec2<u32>(0u, 0u);
    }
    var free_idx = m_sz - 1u;
    out_v[2u * (free_idx * n + eig_idx)] = one_bits;
    if (free_idx > 0u) {
      var r = i32(free_idx) - 1;
      loop {
        if (r < 0) { break; }
        let ur = u32(r);
        var sum = cdf64_zero();
        for (var c = ur + 1u; c <= free_idx; c = c + 1u) {
          let m_rc = get_m_c(n, ur, c);
          let uc_re = unpack_f64_df64(out_v[2u * (c * n + eig_idx)]);
          let uc_im = unpack_f64_df64(out_v[2u * (c * n + eig_idx) + 1u]);
          let uc = vec4<f32>(uc_re.x, uc_re.y, uc_im.x, uc_im.y);
          sum = cdf64_add(sum, cdf64_mul(m_rc, uc));
        }
        var denom = get_m_c(n, ur, ur);
        if (cdf64_abs2(denom).x <= 1e-24) {
          denom = vec4<f32>(1e-12, 0.0, 0.0, 0.0);
        }
        let val_u = cdf64_div(cdf64_neg(sum), denom);
        out_v[2u * (ur * n + eig_idx)] = pack_df64_f64(vec2<f32>(val_u.x, val_u.y));
        out_v[2u * (ur * n + eig_idx) + 1u] = pack_df64_f64(vec2<f32>(val_u.z, val_u.w));
        r = r - 1;
      }
    }

    // Transform u by Q: v = Q * u (store temporarily in column 0 of work_m).
    var norm_sq = df64_zero();
    for (var r = 0u; r < n; r = r + 1u) {
      var acc = cdf64_zero();
      for (var c = 0u; c <= free_idx; c = c + 1u) {
        let q_rc = get_q(n, r, c);
        let uc_re = unpack_f64_df64(out_v[2u * (c * n + eig_idx)]);
        let uc_im = unpack_f64_df64(out_v[2u * (c * n + eig_idx) + 1u]);
        let uc = vec4<f32>(uc_re.x, uc_re.y, uc_im.x, uc_im.y);
        acc = cdf64_add(acc, cdf64_scale(uc, q_rc));
      }
      set_m_c(n, r, 0u, acc);
      norm_sq = df64_add(norm_sq, cdf64_abs2(acc));
    }
    let inv_norm = df64_div(one, df64_sqrt(norm_sq));
    for (var r = 0u; r < n; r = r + 1u) {
      let vr = cdf64_scale(get_m_c(n, r, 0u), inv_norm);
      out_v[2u * (r * n + eig_idx)] = pack_df64_f64(vec2<f32>(vr.x, vr.y));
      out_v[2u * (r * n + eig_idx) + 1u] = pack_df64_f64(vec2<f32>(vr.z, vr.w));
    }
  }
}
''';

const String _svdF32Shader = '''
struct SvdParams {
  m: u32,
  n: u32,
  u_cols: u32,
  vt_rows: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> work_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> work_u_full: array<f32>;
@group(0) @binding(3) var<storage, read_write> work_v_full: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_u: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_s: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_vt: array<f32>;
@group(0) @binding(7) var<uniform> params: SvdParams;

@compute @workgroup_size(1)
fn main() {
  let m = params.m;
  let n = params.n;
  let transposed = m < n;
  let p = max(m, n);
  let q = min(m, n);

  for (var r = 0u; r < p; r = r + 1u) {
    for (var c = 0u; c < q; c = c + 1u) {
      if (!transposed) {
        work_b[r * q + c] = in_a[r * n + c];
      } else {
        work_b[r * q + c] = in_a[c * n + r];
      }
    }
  }

  for (var r = 0u; r < q; r = r + 1u) {
    for (var c = 0u; c < q; c = c + 1u) {
      work_v_full[r * q + c] = select(0.0, 1.0, r == c);
    }
  }

  for (var sweep = 0u; sweep < 35u; sweep = sweep + 1u) {
    var max_off = 0.0;
    for (var i = 0u; i < q; i = i + 1u) {
      for (var j = i + 1u; j < q; j = j + 1u) {
        var app = 0.0;
        var aqq = 0.0;
        var apq = 0.0;
        for (var r = 0u; r < p; r = r + 1u) {
          let bri = work_b[r * q + i];
          let brj = work_b[r * q + j];
          app = app + bri * bri;
          aqq = aqq + brj * brj;
          apq = apq + bri * brj;
        }
        let abs_apq = abs(apq);
        let scale_ij = sqrt(max(0.0, app * aqq));
        if (abs_apq > 1e-7 * scale_ij && abs_apq > 0.0) {
          if (abs_apq > max_off) {
            max_off = abs_apq;
          }
          let diff = aqq - app;
          let tau = diff / (2.0 * apq);
          let abs_tau = abs(tau);
          var t_mag = 0.0;
          if (abs_tau > 1e4) {
            t_mag = 1.0 / (2.0 * abs_tau);
          } else {
            t_mag = 1.0 / (abs_tau + sqrt(1.0 + tau * tau));
          }
          let t = select(-t_mag, t_mag, tau >= 0.0);
          let c_rot = 1.0 / sqrt(1.0 + t * t);
          let s_rot = t * c_rot;

          for (var r = 0u; r < p; r = r + 1u) {
            let bri = work_b[r * q + i];
            let brj = work_b[r * q + j];
            work_b[r * q + i] = c_rot * bri - s_rot * brj;
            work_b[r * q + j] = s_rot * bri + c_rot * brj;
          }
          for (var r = 0u; r < q; r = r + 1u) {
            let vri = work_v_full[r * q + i];
            let vrj = work_v_full[r * q + j];
            work_v_full[r * q + i] = c_rot * vri - s_rot * vrj;
            work_v_full[r * q + j] = s_rot * vri + c_rot * vrj;
          }
        }
      }
    }
    if (max_off == 0.0) {
      break;
    }
  }

  for (var j = 0u; j < q; j = j + 1u) {
    var sum_sq = 0.0;
    for (var r = 0u; r < p; r = r + 1u) {
      let brj = work_b[r * q + j];
      sum_sq = sum_sq + brj * brj;
    }
    out_s[j] = sqrt(sum_sq);
  }

  for (var i = 0u; i < q; i = i + 1u) {
    var max_idx = i;
    var max_val = out_s[i];
    for (var j = i + 1u; j < q; j = j + 1u) {
      let sj = out_s[j];
      if (sj > max_val) {
        max_val = sj;
        max_idx = j;
      }
    }
    if (max_idx != i) {
      let tmp_s = out_s[i];
      out_s[i] = out_s[max_idx];
      out_s[max_idx] = tmp_s;
      for (var r = 0u; r < p; r = r + 1u) {
        let tmp_b = work_b[r * q + i];
        work_b[r * q + i] = work_b[r * q + max_idx];
        work_b[r * q + max_idx] = tmp_b;
      }
      for (var r = 0u; r < q; r = r + 1u) {
        let tmp_v = work_v_full[r * q + i];
        work_v_full[r * q + i] = work_v_full[r * q + max_idx];
        work_v_full[r * q + max_idx] = tmp_v;
      }
    }
  }

  let s0 = out_s[0];
  let tol = max(1e-6 * s0, 1e-30);
  var rank_count = 0u;
  for (var j = 0u; j < q; j = j + 1u) {
    let sj = out_s[j];
    if (sj > tol) {
      for (var r = 0u; r < p; r = r + 1u) {
        work_u_full[r * p + j] = work_b[r * q + j] / sj;
      }
      rank_count = j + 1u;
    } else {
      break;
    }
  }

  for (var col = rank_count; col < p; col = col + 1u) {
    for (var cand = 0u; cand < p; cand = cand + 1u) {
      for (var r = 0u; r < p; r = r + 1u) {
        work_u_full[r * p + col] = select(0.0, 1.0, r == cand);
      }
      for (var gs_pass = 0u; gs_pass < 2u; gs_pass = gs_pass + 1u) {
        for (var prev = 0u; prev < col; prev = prev + 1u) {
          var dot_val = 0.0;
          for (var r = 0u; r < p; r = r + 1u) {
            dot_val = dot_val + work_u_full[r * p + prev] * work_u_full[r * p + col];
          }
          for (var r = 0u; r < p; r = r + 1u) {
            work_u_full[r * p + col] = work_u_full[r * p + col] - dot_val * work_u_full[r * p + prev];
          }
        }
      }
      var norm_sq = 0.0;
      for (var r = 0u; r < p; r = r + 1u) {
        let ur_col = work_u_full[r * p + col];
        norm_sq = norm_sq + ur_col * ur_col;
      }
      if (norm_sq > 0.1) {
        let inv_norm = 1.0 / sqrt(norm_sq);
        for (var r = 0u; r < p; r = r + 1u) {
          work_u_full[r * p + col] = work_u_full[r * p + col] * inv_norm;
        }
        break;
      }
    }
  }

  let u_cols = params.u_cols;
  let vt_rows = params.vt_rows;
  if (!transposed) {
    for (var r = 0u; r < m; r = r + 1u) {
      for (var c = 0u; c < u_cols; c = c + 1u) {
        out_u[r * u_cols + c] = work_u_full[r * p + c];
      }
    }
    for (var r = 0u; r < vt_rows; r = r + 1u) {
      for (var c = 0u; c < n; c = c + 1u) {
        out_vt[r * n + c] = work_v_full[c * q + r];
      }
    }
  } else {
    for (var r = 0u; r < m; r = r + 1u) {
      for (var c = 0u; c < u_cols; c = c + 1u) {
        out_u[r * u_cols + c] = work_v_full[r * q + c];
      }
    }
    for (var r = 0u; r < vt_rows; r = r + 1u) {
      for (var c = 0u; c < n; c = c + 1u) {
        out_vt[r * n + c] = work_u_full[c * p + r];
      }
    }
  }
}
''';

const String _eigF32Shader = '''
struct EigParams {
  n: u32,
  compute_vectors: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> work_h: array<f32>;
@group(0) @binding(2) var<storage, read_write> work_q: array<f32>;
@group(0) @binding(3) var<storage, read_write> work_m: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> out_w: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> out_v: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> params: EigParams;

fn c64_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn c64_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let denom = b.x * b.x + b.y * b.y;
  return vec2<f32>(
    (a.x * b.x + a.y * b.y) / denom,
    (a.y * b.x - a.x * b.y) / denom
  );
}

fn c64_abs2(a: vec2<f32>) -> f32 {
  return a.x * a.x + a.y * a.y;
}

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  for (var r = 0u; r < n; r = r + 1u) {
    for (var c = 0u; c < n; c = c + 1u) {
      work_h[r * n + c] = in_a[r * n + c];
      work_q[r * n + c] = select(0.0, 1.0, r == c);
    }
  }

  if (n > 2u) {
    for (var k = 0u; k < n - 2u; k = k + 1u) {
      var norm_sq = 0.0;
      for (var i = k + 1u; i < n; i = i + 1u) {
        let hik = work_h[i * n + k];
        norm_sq = norm_sq + hik * hik;
      }
      let norm_x = sqrt(norm_sq);
      if (norm_x > 0.0) {
        let x0 = work_h[(k + 1u) * n + k];
        let alpha = select(norm_x, -norm_x, x0 >= 0.0);
        let v0 = x0 - alpha;
        var tail_sq = 0.0;
        for (var i = k + 2u; i < n; i = i + 1u) {
          let vi = work_h[i * n + k];
          tail_sq = tail_sq + vi * vi;
        }
        let v_norm_sq = v0 * v0 + tail_sq;
        if (v_norm_sq > 0.0) {
          work_h[(k + 1u) * n + k] = v0;
          for (var c = k + 1u; c < n; c = c + 1u) {
            var dot_v = 0.0;
            for (var i = k + 1u; i < n; i = i + 1u) {
              dot_v = dot_v + work_h[i * n + k] * work_h[i * n + c];
            }
            let scale = (2.0 * dot_v) / v_norm_sq;
            for (var i = k + 1u; i < n; i = i + 1u) {
              work_h[i * n + c] = work_h[i * n + c] - scale * work_h[i * n + k];
            }
          }
          for (var r = 0u; r < n; r = r + 1u) {
            var dot_v = 0.0;
            for (var i = k + 1u; i < n; i = i + 1u) {
              dot_v = dot_v + work_h[r * n + i] * work_h[i * n + k];
            }
            let scale = (2.0 * dot_v) / v_norm_sq;
            for (var i = k + 1u; i < n; i = i + 1u) {
              work_h[r * n + i] = work_h[r * n + i] - scale * work_h[i * n + k];
            }
          }
          for (var r = 0u; r < n; r = r + 1u) {
            var dot_v = 0.0;
            for (var i = k + 1u; i < n; i = i + 1u) {
              dot_v = dot_v + work_q[r * n + i] * work_h[i * n + k];
            }
            let scale = (2.0 * dot_v) / v_norm_sq;
            for (var i = k + 1u; i < n; i = i + 1u) {
              work_q[r * n + i] = work_q[r * n + i] - scale * work_h[i * n + k];
            }
          }
          work_h[(k + 1u) * n + k] = alpha;
          for (var i = k + 2u; i < n; i = i + 1u) {
            work_h[i * n + k] = 0.0;
          }
        }
      }
    }
  }

  if (n > 1u) {
    var active_end = i32(n) - 1;
    var iter = 0u;
    loop {
      if (active_end <= 0 || iter >= 300u) {
        break;
      }
      var l = active_end;
      loop {
        if (l <= 0) { break; }
        let sub = abs(work_h[u32(l) * n + u32(l - 1)]);
        let diag_sum = abs(work_h[u32(l - 1) * n + u32(l - 1)]) + abs(work_h[u32(l) * n + u32(l)]);
        if (sub <= 1e-6 * max(1e-12, diag_sum)) {
          work_h[u32(l) * n + u32(l - 1)] = 0.0;
          break;
        }
        l = l - 1;
      }
      if (l == active_end) {
        active_end = active_end - 1;
        continue;
      }
      if (l == active_end - 1) {
        let p_idx = u32(active_end - 1);
        let q_idx = u32(active_end);
        let a = work_h[p_idx * n + p_idx];
        let b = work_h[p_idx * n + q_idx];
        let c = work_h[q_idx * n + p_idx];
        let d = work_h[q_idx * n + q_idx];
        let half_diff = 0.5 * (a - d);
        let disc = half_diff * half_diff + b * c;
        if (disc >= 0.0) {
          let root = sqrt(disc);
          let shift = d + select(half_diff + root, half_diff - root, half_diff < 0.0);
          var vx = a - shift;
          var vy = c;
          if (abs(b) > abs(c)) {
            vx = b;
            vy = shift - a;
          }
          let r_len = sqrt(vx * vx + vy * vy);
          if (r_len > 0.0) {
            let cs = vx / r_len;
            let sn = vy / r_len;
            for (var col = p_idx; col < n; col = col + 1u) {
              let hp = work_h[p_idx * n + col];
              let hq = work_h[q_idx * n + col];
              work_h[p_idx * n + col] = cs * hp + sn * hq;
              work_h[q_idx * n + col] = cs * hq - sn * hp;
            }
            for (var row = 0u; row <= q_idx; row = row + 1u) {
              let hp = work_h[row * n + p_idx];
              let hq = work_h[row * n + q_idx];
              work_h[row * n + p_idx] = cs * hp + sn * hq;
              work_h[row * n + q_idx] = cs * hq - sn * hp;
            }
            for (var row = 0u; row < n; row = row + 1u) {
              let qp = work_q[row * n + p_idx];
              let qq = work_q[row * n + q_idx];
              work_q[row * n + p_idx] = cs * qp + sn * qq;
              work_q[row * n + q_idx] = cs * qq - sn * qp;
            }
          }
          work_h[q_idx * n + p_idx] = 0.0;
        }
        active_end = active_end - 2;
        continue;
      }

      iter = iter + 1u;
      let p_idx = u32(active_end - 1);
      let q_idx = u32(active_end);
      let a = work_h[p_idx * n + p_idx];
      let b = work_h[p_idx * n + q_idx];
      let c = work_h[q_idx * n + p_idx];
      let d = work_h[q_idx * n + q_idx];
      let half_diff = 0.5 * (a - d);
      let disc = half_diff * half_diff + b * c;
      var shift = d;
      if (disc >= 0.0) {
        let root = sqrt(disc);
        shift = d + select(half_diff - root, half_diff + root, half_diff >= 0.0);
      } else if ((iter % 10u) == 0u) {
        shift = d + abs(c);
      }

      let u_start = u32(l);
      let u_end = u32(active_end);
      for (var i = u_start; i <= u_end; i = i + 1u) {
        work_h[i * n + i] = work_h[i * n + i] - shift;
      }
      for (var i = u_start; i < u_end; i = i + 1u) {
        let x_val = work_h[i * n + i];
        let y_val = work_h[(i + 1u) * n + i];
        let r_len = sqrt(x_val * x_val + y_val * y_val);
        var cs = 1.0;
        var sn = 0.0;
        if (r_len > 0.0) {
          cs = x_val / r_len;
          sn = y_val / r_len;
        }
        out_w[i] = vec2<f32>(cs, sn);
        for (var col = i; col < n; col = col + 1u) {
          let hi = work_h[i * n + col];
          let hi1 = work_h[(i + 1u) * n + col];
          work_h[i * n + col] = cs * hi + sn * hi1;
          work_h[(i + 1u) * n + col] = cs * hi1 - sn * hi;
        }
        work_h[(i + 1u) * n + i] = 0.0;
      }
      for (var i = u_start; i < u_end; i = i + 1u) {
        let cs = out_w[i].x;
        let sn = out_w[i].y;
        for (var row = 0u; row <= i + 1u; row = row + 1u) {
          let hi = work_h[row * n + i];
          let hi1 = work_h[row * n + i + 1u];
          work_h[row * n + i] = cs * hi + sn * hi1;
          work_h[row * n + i + 1u] = cs * hi1 - sn * hi;
        }
        for (var row = 0u; row < n; row = row + 1u) {
          let qi = work_q[row * n + i];
          let qi1 = work_q[row * n + i + 1u];
          work_q[row * n + i] = cs * qi + sn * qi1;
          work_q[row * n + i + 1u] = cs * qi1 - sn * qi;
        }
      }
      for (var i = u_start; i <= u_end; i = i + 1u) {
        work_h[i * n + i] = work_h[i * n + i] + shift;
      }
    }
  }

  var idx = 0u;
  loop {
    if (idx >= n) { break; }
    if (idx + 1u < n) {
      let sub = work_h[(idx + 1u) * n + idx];
      if (abs(sub) > 1e-6) {
        let a = work_h[idx * n + idx];
        let b = work_h[idx * n + idx + 1u];
        let c = sub;
        let d = work_h[(idx + 1u) * n + idx + 1u];
        let mean = 0.5 * (a + d);
        let half_diff = 0.5 * (a - d);
        let disc = half_diff * half_diff + b * c;
        let imag = sqrt(abs(disc));
        out_w[idx] = vec2<f32>(mean, imag);
        out_w[idx + 1u] = vec2<f32>(mean, -imag);
        idx = idx + 2u;
        continue;
      }
    }
    out_w[idx] = vec2<f32>(work_h[idx * n + idx], 0.0);
    idx = idx + 1u;
  }

  if (params.compute_vectors == 0u) {
    return;
  }

  for (var eig_idx = 0u; eig_idx < n; eig_idx = eig_idx + 1u) {
    let lambda = out_w[eig_idx];
    var m_end = eig_idx;
    if (eig_idx + 1u < n && abs(work_h[(eig_idx + 1u) * n + eig_idx]) > 1e-6) {
      m_end = eig_idx + 1u;
    }
    let m_sz = m_end + 1u;

    for (var r = 0u; r < m_sz; r = r + 1u) {
      for (var c = 0u; c < m_sz; c = c + 1u) {
        var val = vec2<f32>(work_h[r * n + c], 0.0);
        if (r == c) {
          val = val - lambda;
        }
        work_m[r * n + c] = val;
      }
    }

    if (m_sz > 1u) {
      for (var k = 0u; k < m_sz - 1u; k = k + 1u) {
        let diag_k = work_m[k * n + k];
        let sub_k = work_m[(k + 1u) * n + k];
        if (c64_abs2(sub_k) > c64_abs2(diag_k)) {
          for (var c = k; c < m_sz; c = c + 1u) {
            let tmp = work_m[k * n + c];
            work_m[k * n + c] = work_m[(k + 1u) * n + c];
            work_m[(k + 1u) * n + c] = tmp;
          }
        }
        let piv = work_m[k * n + k];
        let elim = work_m[(k + 1u) * n + k];
        if (c64_abs2(piv) > 1e-20 && c64_abs2(elim) > 0.0) {
          let factor = c64_div(elim, piv);
          work_m[(k + 1u) * n + k] = vec2<f32>(0.0, 0.0);
          for (var c = k + 1u; c < m_sz; c = c + 1u) {
            work_m[(k + 1u) * n + c] = work_m[(k + 1u) * n + c] - c64_mul(factor, work_m[k * n + c]);
          }
        }
      }
    }

    for (var i = 0u; i < n; i = i + 1u) {
      out_v[i * n + eig_idx] = vec2<f32>(0.0, 0.0);
    }
    let free_idx = m_sz - 1u;
    out_v[free_idx * n + eig_idx] = vec2<f32>(1.0, 0.0);
    if (free_idx > 0u) {
      var r = i32(free_idx) - 1;
      loop {
        if (r < 0) { break; }
        let ur = u32(r);
        var sum = vec2<f32>(0.0, 0.0);
        for (var c = ur + 1u; c <= free_idx; c = c + 1u) {
          sum = sum + c64_mul(work_m[ur * n + c], out_v[c * n + eig_idx]);
        }
        var denom = work_m[ur * n + ur];
        if (c64_abs2(denom) <= 1e-16) {
          denom = vec2<f32>(1e-8, 0.0);
        }
        out_v[ur * n + eig_idx] = c64_div(-sum, denom);
        r = r - 1;
      }
    }

    var norm_sq = 0.0;
    for (var r = 0u; r < n; r = r + 1u) {
      var acc = vec2<f32>(0.0, 0.0);
      for (var c = 0u; c <= free_idx; c = c + 1u) {
        acc = acc + out_v[c * n + eig_idx] * work_q[r * n + c];
      }
      work_m[r * n] = acc;
      norm_sq = norm_sq + c64_abs2(acc);
    }
    let inv_norm = 1.0 / sqrt(max(1e-30, norm_sq));
    for (var r = 0u; r < n; r = r + 1u) {
      out_v[r * n + eig_idx] = work_m[r * n] * inv_norm;
    }
  }
}
''';

/// Dispatches the Singular Value Decomposition kernel on [device].
({GpuBuffer u, GpuBuffer s, GpuBuffer vt}) dispatchSvdGpu(
  GpuDevice device,
  GpuBuffer inputBuffer,
  int m,
  int n, {
  required bool fullMatrices,
  bool singlePrecision = false,
}) {
  final elemBytes = singlePrecision ? 4 : 8;
  final p = math.max(m, n);
  final q = math.min(m, n);
  final uCols = fullMatrices ? m : q;
  final vtRows = fullMatrices ? n : q;

  final workB = device.createBuffer(
    sizeInBytes: math.max(1, p * q) * elemBytes,
  );
  final workUFull = device.createBuffer(
    sizeInBytes: math.max(1, p * p) * elemBytes,
  );
  final workVFull = device.createBuffer(
    sizeInBytes: math.max(1, q * q) * elemBytes,
  );
  final outU = device.createBuffer(
    sizeInBytes: math.max(1, m * uCols) * elemBytes,
  );
  final outS = device.createBuffer(sizeInBytes: math.max(1, q) * elemBytes);
  final outVt = device.createBuffer(
    sizeInBytes: math.max(1, vtRows * n) * elemBytes,
  );
  if (m == 0 || n == 0) return (u: outU, s: outS, vt: outVt);

  final module = singlePrecision
      ? getOrCreateLinalgShader('linalg_svd_f32', () => _svdF32Shader)
      : getOrCreateLinalgShader('linalg_svd_f64', () => _svdRealShader);
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputBuffer, workB, workUFull, workVFull, outU, outS, outVt],
    uniforms: [m, n, uCols, vtRows],
    workgroupsX: 1,
  );
  return (u: outU, s: outS, vt: outVt);
}

/// Dispatches the general non-symmetric eigendecomposition kernel on [device].
({GpuBuffer eigenvalues, GpuBuffer eigenvectors}) dispatchEigGpu(
  GpuDevice device,
  GpuBuffer inputBuffer,
  int n, {
  required bool computeVectors,
  bool singlePrecision = false,
}) {
  final realBytes = singlePrecision ? 4 : 8;
  final complexBytes = singlePrecision ? 8 : 16;
  final workH = device.createBuffer(
    sizeInBytes: math.max(1, n * n) * realBytes,
  );
  final workQ = device.createBuffer(
    sizeInBytes: math.max(1, n * n) * realBytes,
  );
  final workM = device.createBuffer(
    sizeInBytes: math.max(1, n * n) * complexBytes,
  );
  final outW = device.createBuffer(sizeInBytes: math.max(1, n) * complexBytes);
  final outV = device.createBuffer(
    sizeInBytes: math.max(1, n * n) * complexBytes,
  );
  if (n == 0) return (eigenvalues: outW, eigenvectors: outV);

  final module = singlePrecision
      ? getOrCreateLinalgShader('linalg_eig_f32', () => _eigF32Shader)
      : getOrCreateLinalgShader('linalg_eig_f64', () => _eigRealShader);
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: [inputBuffer, workH, workQ, workM, outW, outV],
    uniforms: [n, computeVectors ? 1 : 0, 0, 0],
    workgroupsX: 1,
  );
  return (eigenvalues: outW, eigenvectors: outV);
}
