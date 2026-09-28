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

import 'wgsl_types.dart';

/// Pre-defined WGSL compute shader templates and code generators.
final class WgslTemplates {
  WgslTemplates._();

  /// Standard WGSL header for strided multi-index translation.
  static const String stridedHeader = '''
struct StridedMetadata {
  total_elements: u32,
  rank: u32,
  pad0: u32,
  pad1: u32,
  shape: array<vec4<u32>, 2>,      // Up to 8 dimensions (2 vec4s)
  strides_a: array<vec4<u32>, 2>,  // Up to 8 dimensions
  strides_b: array<vec4<u32>, 2>,  // Up to 8 dimensions
  strides_out: array<vec4<u32>, 2>,// Up to 8 dimensions
  offset_a: u32,
  offset_b: u32,
  offset_out: u32,
  scalar_param: f32,
}

fn get_shape_dim(meta: StridedMetadata, dim: u32) -> u32 {
  if (dim < 4u) {
    return meta.shape[0][dim];
  } else {
    return meta.shape[1][dim - 4u];
  }
}

fn get_stride_a(meta: StridedMetadata, dim: u32) -> u32 {
  if (dim < 4u) {
    return meta.strides_a[0][dim];
  } else {
    return meta.strides_a[1][dim - 4u];
  }
}

fn get_stride_b(meta: StridedMetadata, dim: u32) -> u32 {
  if (dim < 4u) {
    return meta.strides_b[0][dim];
  } else {
    return meta.strides_b[1][dim - 4u];
  }
}

fn get_stride_out(meta: StridedMetadata, dim: u32) -> u32 {
  if (dim < 4u) {
    return meta.strides_out[0][dim];
  } else {
    return meta.strides_out[1][dim - 4u];
  }
}

fn flat_to_strided_offsets(
  idx: u32,
  meta: StridedMetadata,
  out_offset_a: ptr<function, u32>,
  out_offset_b: ptr<function, u32>,
  out_offset_dst: ptr<function, u32>
) {
  var rem = idx;
  var off_a = meta.offset_a;
  var off_b = meta.offset_b;
  var off_dst = meta.offset_out;

  for (var d = i32(meta.rank) - 1; d >= 0; d--) {
    let dim_size = get_shape_dim(meta, u32(d));
    if (dim_size > 0u) {
      let coord = rem % dim_size;
      rem = rem / dim_size;
      off_a += coord * get_stride_a(meta, u32(d));
      off_b += coord * get_stride_b(meta, u32(d));
      off_dst += coord * get_stride_out(meta, u32(d));
    }
  }

  *out_offset_a = off_a;
  *out_offset_b = off_b;
  *out_offset_dst = off_dst;
}
''';

  /// Standard WGSL activation / math helper functions.
  static const String mathHelpers = '''
fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

fn gelu(x: f32) -> f32 {
  return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn swish(x: f32) -> f32 {
  return silu(x);
}

fn hardswish(x: f32) -> f32 {
  return x * clamp(x + 3.0, 0.0, 6.0) / 6.0;
}

fn softplus(x: f32) -> f32 {
  return log(1.0 + exp(x));
}

fn mish(x: f32) -> f32 {
  return x * tanh(log(1.0 + exp(x)));
}
''';

  /// Maps a standard binary operator name to its corresponding WGSL expression snippet.
  static String getWgslOpExpression(String op, String a, String b) {
    switch (op.toLowerCase()) {
      case 'add':
      case '+':
        return '$a + $b';
      case 'sub':
      case 'subtract':
      case '-':
        return '$a - $b';
      case 'mul':
      case 'multiply':
      case '*':
        return '$a * $b';
      case 'div':
      case 'divide':
      case '/':
        return '$a / $b';
      case 'pow':
      case 'power':
        return 'pow($a, $b)';
      case 'rem':
      case 'remainder':
      case 'mod':
      case '%':
        return '$a % $b';
      case 'max':
      case 'maximum':
        return 'max($a, $b)';
      case 'min':
      case 'minimum':
        return 'min($a, $b)';
      case 'eq':
      case 'equal':
      case '==':
        return 'select(0.0, 1.0, $a == $b)';
      case 'neq':
      case 'notequal':
      case '!=':
        return 'select(0.0, 1.0, $a != $b)';
      case 'gt':
      case 'greater':
      case '>':
        return 'select(0.0, 1.0, $a > $b)';
      case 'lt':
      case 'less':
      case '<':
        return 'select(0.0, 1.0, $a < $b)';
      case 'gte':
      case 'greaterequal':
      case '>=':
        return 'select(0.0, 1.0, $a >= $b)';
      case 'lte':
      case 'lessequal':
      case '<=':
        return 'select(0.0, 1.0, $a <= $b)';
      case 'and':
      case '&':
        return 'select(0.0, 1.0, ($a > 0.0) && ($b > 0.0))';
      case 'or':
      case '|':
        return 'select(0.0, 1.0, ($a > 0.0) || ($b > 0.0))';
      case 'atan2':
        return 'atan2($a, $b)';
      case 'hypot':
        return 'sqrt(($a * $a) + ($b * $b))';
      case 'step':
        return 'step($a, $b)';
      default:
        throw ArgumentError('Unsupported binary operation: $op');
    }
  }

  /// Maps a standard unary operator name to its corresponding WGSL expression snippet.
  static String getWgslUnaryExpression(String op, String x) {
    switch (op.toLowerCase()) {
      case 'relu':
        return 'max($x, 0.0)';
      case 'silu':
        return 'silu($x)';
      case 'gelu':
        return 'gelu($x)';
      case 'sigmoid':
        return 'sigmoid($x)';
      case 'tanh':
        return 'tanh($x)';
      case 'exp':
        return 'exp($x)';
      case 'log':
        return 'log($x)';
      case 'sqrt':
        return 'sqrt($x)';
      case 'rsqrt':
      case 'inversesqrt':
        return 'inverseSqrt($x)';
      case 'abs':
        return 'abs($x)';
      case 'negate':
      case 'neg':
      case '-':
        return '-$x';
      case 'sin':
        return 'sin($x)';
      case 'cos':
        return 'cos($x)';
      case 'tan':
        return 'tan($x)';
      case 'asin':
        return 'asin($x)';
      case 'acos':
        return 'acos($x)';
      case 'atan':
        return 'atan($x)';
      case 'sinh':
        return 'sinh($x)';
      case 'cosh':
        return 'cosh($x)';
      case 'floor':
        return 'floor($x)';
      case 'ceil':
        return 'ceil($x)';
      case 'round':
        return 'round($x)';
      case 'reciprocal':
        return '1.0 / $x';
      case 'hardswish':
        return 'hardswish($x)';
      case 'softplus':
        return 'softplus($x)';
      case 'mish':
        return 'mish($x)';
      case 'not':
      case '~':
      case '!':
        return 'select(0.0, 1.0, $x <= 0.0)';
      case 'fract':
        return 'fract($x)';
      case 'sign':
        return 'sign($x)';
      default:
        throw ArgumentError('Unsupported unary operation: $op');
    }
  }

  /// Maps a reduction operation name to its combination operator.
  static String getWgslReductionOp(String op, String a, String b) {
    switch (op.toLowerCase()) {
      case 'sum':
      case 'mean':
      case 'sum_sq':
        return '$a + $b';
      case 'prod':
      case 'product':
        return '$a * $b';
      case 'min':
        return 'min($a, $b)';
      case 'max':
        return 'max($a, $b)';
      default:
        throw ArgumentError('Unsupported reduction operation: $op');
    }
  }

  /// Initial identity value for a reduction operation.
  static String getWgslReductionInit(String op, WgslDType dtype) {
    final lowerOp = op.toLowerCase();
    switch (dtype) {
      case WgslDType.float16:
        switch (lowerOp) {
          case 'sum':
          case 'mean':
          case 'sum_sq':
            return '0.0h';
          case 'prod':
          case 'product':
            return '1.0h';
          case 'min':
            return '65504.0h';
          case 'max':
            return '-65504.0h';
          default:
            return '0.0h';
        }
      case WgslDType.int32:
        switch (lowerOp) {
          case 'sum':
          case 'mean':
          case 'sum_sq':
            return '0';
          case 'prod':
          case 'product':
            return '1';
          case 'min':
            return '2147483647';
          case 'max':
            return '-2147483648';
          default:
            return '0';
        }
      case WgslDType.uint32:
        switch (lowerOp) {
          case 'sum':
          case 'mean':
          case 'sum_sq':
            return '0u';
          case 'prod':
          case 'product':
            return '1u';
          case 'min':
            return '4294967295u';
          case 'max':
            return '0u';
          default:
            return '0u';
        }
      default:
        switch (lowerOp) {
          case 'sum':
          case 'mean':
          case 'sum_sq':
            return '0.0';
          case 'prod':
          case 'product':
            return '1.0';
          case 'min':
            return '3.402823e+38'; // f32 max
          case 'max':
            return '-3.402823e+38'; // f32 min
          default:
            return '0.0';
        }
    }
  }

  /// Generates an elementwise binary compute shader (e.g. add, sub, mul, div, pow).
  static WgslShaderModule elementwiseBinary({
    required String op,
    WgslDType dtype = WgslDType.float32,
    bool strided = false,
    int workgroupSize = 256,
  }) {
    final opExpr = getWgslOpExpression(op, 'a_val', 'b_val');
    final wgSize = WgslWorkgroupSize(workgroupSize, 1, 1);

    final String code;
    final List<WgslBinding> bindings;

    if (!strided) {
      // Contiguous 1D fast path
      bindings = [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src_a',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'src_b',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst',
          dtype: dtype,
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 3,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'Uniforms',
        ),
      ];

      code =
          '''
// WGSL Elementwise Binary: $op (Contiguous)
struct Uniforms {
  total_elements: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

${bindings[0].toWgslDeclaration()}
${bindings[1].toWgslDeclaration()}
${bindings[2].toWgslDeclaration()}
${bindings[3].toWgslDeclaration()}

$mathHelpers

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  var idx = global_id.x;
  let stride = num_workgroups.x * ${workgroupSize}u;
  while (idx < uniforms.total_elements) {
    let a_val = src_a[idx];
    let b_val = src_b[idx];
    dst[idx] = $opExpr;
    idx += stride;
  }
}
''';
    } else {
      // Strided multidimensional layout with broadcasting support
      bindings = [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src_a',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'src_b',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'dst',
          dtype: dtype,
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 3,
          name: 'meta',
          isUniform: true,
          customTypeName: 'StridedMetadata',
        ),
      ];

      code =
          '''
// WGSL Elementwise Binary: $op (Strided & Broadcast)
$stridedHeader

${bindings[0].toWgslDeclaration()}
${bindings[1].toWgslDeclaration()}
${bindings[2].toWgslDeclaration()}
${bindings[3].toWgslDeclaration()}

$mathHelpers

@compute ${wgSize.toAttribute()}
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= meta.total_elements) {
    return;
  }

  var off_a: u32 = 0u;
  var off_b: u32 = 0u;
  var off_dst: u32 = 0u;
  flat_to_strided_offsets(idx, meta, &off_a, &off_b, &off_dst);

  let a_val = src_a[off_a];
  let b_val = src_b[off_b];
  dst[off_dst] = $opExpr;
}
''';
    }

    return WgslShaderModule(
      name: 'elementwise_binary_${op}_${strided ? "strided" : "contiguous"}',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {'op': op, 'dtype': dtype.wgslType, 'strided': strided},
    );
  }

  /// Generates an elementwise unary compute shader (e.g. relu, silu, gelu, exp, log, tanh).
  static WgslShaderModule elementwiseUnary({
    required String op,
    WgslDType dtype = WgslDType.float32,
    bool strided = false,
    int workgroupSize = 256,
  }) {
    final unaryExpr = getWgslUnaryExpression(op, 'x_val');
    final wgSize = WgslWorkgroupSize(workgroupSize, 1, 1);

    final String code;
    final List<WgslBinding> bindings;

    if (!strided) {
      bindings = [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: dtype,
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'Uniforms',
        ),
      ];

      code =
          '''
// WGSL Elementwise Unary: $op (Contiguous)
struct Uniforms {
  total_elements: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

${bindings[0].toWgslDeclaration()}
${bindings[1].toWgslDeclaration()}
${bindings[2].toWgslDeclaration()}

$mathHelpers

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  var idx = global_id.x;
  let stride = num_workgroups.x * ${workgroupSize}u;
  while (idx < uniforms.total_elements) {
    let x_val = src[idx];
    dst[idx] = $unaryExpr;
    idx += stride;
  }
}
''';
    } else {
      bindings = [
        WgslBinding(
          group: 0,
          binding: 0,
          name: 'src',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
        WgslBinding(
          group: 0,
          binding: 1,
          name: 'dst',
          dtype: dtype,
          access: WgslBufferAccess.readWrite,
        ),
        WgslBinding(
          group: 0,
          binding: 2,
          name: 'meta',
          isUniform: true,
          customTypeName: 'StridedMetadata',
        ),
      ];

      code =
          '''
// WGSL Elementwise Unary: $op (Strided)
$stridedHeader

${bindings[0].toWgslDeclaration()}
${bindings[1].toWgslDeclaration()}
${bindings[2].toWgslDeclaration()}

$mathHelpers

@compute ${wgSize.toAttribute()}
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= meta.total_elements) {
    return;
  }

  var off_a: u32 = 0u;
  var off_b: u32 = 0u;
  var off_dst: u32 = 0u;
  flat_to_strided_offsets(idx, meta, &off_a, &off_b, &off_dst);

  let x_val = src[off_a];
  dst[off_dst] = $unaryExpr;
}
''';
    }

    return WgslShaderModule(
      name: 'elementwise_unary_${op}_${strided ? "strided" : "contiguous"}',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {'op': op, 'dtype': dtype.wgslType, 'strided': strided},
    );
  }

  /// Generates a parallel tree-reduction WGSL compute shader using workgroup shared memory.
  static WgslShaderModule treeReduction({
    required String op,
    int workgroupSize = 256,
    WgslDType dtype = WgslDType.float32,
  }) {
    final typeName = dtype.wgslType;
    final initVal = getWgslReductionInit(op, dtype);
    final reduceOp = getWgslReductionOp(op, 'sdata[tid]', 'sdata[tid + s]');
    final isSumSq = op.toLowerCase() == 'sum_sq';
    final wgSize = WgslWorkgroupSize(workgroupSize, 1, 1);

    final bindings = [
      WgslBinding(
        group: 0,
        binding: 0,
        name: 'src',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 1,
        name: 'dst',
        dtype: dtype,
        access: WgslBufferAccess.readWrite,
      ),
      WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'ReductionUniforms',
      ),
    ];

    final accumStmt = isSumSq
        ? 'my_val = my_val + elem * elem;'
        : 'my_val = ${getWgslReductionOp(op, "my_val", "elem")};';

    final code =
        '''
// WGSL Parallel Tree Reduction: $op
struct ReductionUniforms {
  total_elements: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

${bindings[0].toWgslDeclaration()}
${bindings[1].toWgslDeclaration()}
${bindings[2].toWgslDeclaration()}

var<workgroup> sdata: array<$typeName, $workgroupSize>;

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let tid = local_id.x;
  var my_val: $typeName = $initVal;

  // Grid-stride loop: accumulate multiple input elements per thread into register
  var i = global_id.x;
  let stride = num_workgroups.x * ${workgroupSize}u;
  while (i < uniforms.total_elements) {
    let elem = src[i];
    $accumStmt
    i += stride;
  }

  sdata[tid] = my_val;
  workgroupBarrier();

  // In-workgroup tree reduction loop
  for (var s = ${workgroupSize ~/ 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sdata[tid] = $reduceOp;
    }
    workgroupBarrier();
  }

  // Workgroup leader writes result to workgroup output slot
  if (tid == 0u) {
    dst[workgroup_id.x] = sdata[0];
  }
}
''';

    return WgslShaderModule(
      name: 'reduction_$op',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {
        'op': op,
        'workgroupSize': workgroupSize,
        'dtype': dtype.wgslType,
      },
    );
  }

  /// Generates a tiled block GEMM (Matrix Multiplication) compute shader with shared memory.
  static WgslShaderModule tiledMatmul({
    int tileSize = 16,
    WgslDType dtype = WgslDType.float32,
    bool strided = true,
    bool hasBias = false,
  }) {
    final typeName = dtype.wgslType;
    final wgSize = WgslWorkgroupSize(tileSize, tileSize, 1);

    final bindings = [
      WgslBinding(
        group: 0,
        binding: 0,
        name: 'matrix_a',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 1,
        name: 'matrix_b',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 2,
        name: 'matrix_c',
        dtype: dtype,
        access: WgslBufferAccess.readWrite,
      ),
      if (hasBias)
        WgslBinding(
          group: 0,
          binding: 3,
          name: 'bias',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
      WgslBinding(
        group: 0,
        binding: hasBias ? 4 : 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'MatmulUniforms',
      ),
    ];

    final code =
        '''
// WGSL Tiled Block GEMM (Matrix Multiplication)
// Computes C = alpha * (A x B) + beta * C (+ bias)
struct MatmulUniforms {
  M: u32,
  N: u32,
  K: u32,
  stride_a_m: u32,
  stride_a_k: u32,
  stride_b_k: u32,
  stride_b_n: u32,
  stride_c_m: u32,
  stride_c_n: u32,
  offset_a: u32,
  offset_b: u32,
  offset_c: u32,
  alpha: f32,
  beta: f32,
  pad0: u32,
  pad1: u32,
}

${bindings.map((b) => b.toWgslDeclaration()).join('\n')}

var<workgroup> tile_a: array<array<$typeName, $tileSize>, $tileSize>;
var<workgroup> tile_b: array<array<$typeName, $tileSize>, $tileSize>;

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = global_id.y;
  let col = global_id.x;
  let tx = local_id.x;
  let ty = local_id.y;

  var acc: $typeName = 0.0;
  let num_tiles = (uniforms.K + ${tileSize}u - 1u) / ${tileSize}u;

  for (var t = 0u; t < num_tiles; t++) {
    let tiled_k_a = t * ${tileSize}u + tx;
    if (row < uniforms.M && tiled_k_a < uniforms.K) {
      let idx_a = uniforms.offset_a + row * uniforms.stride_a_m + tiled_k_a * uniforms.stride_a_k;
      tile_a[ty][tx] = matrix_a[idx_a];
    } else {
      tile_a[ty][tx] = 0.0;
    }

    let tiled_k_b = t * ${tileSize}u + ty;
    if (tiled_k_b < uniforms.K && col < uniforms.N) {
      let idx_b = uniforms.offset_b + tiled_k_b * uniforms.stride_b_k + col * uniforms.stride_b_n;
      tile_b[ty][tx] = matrix_b[idx_b];
    } else {
      tile_b[ty][tx] = 0.0;
    }

    workgroupBarrier();

    for (var k = 0u; k < ${tileSize}u; k++) {
      acc += tile_a[ty][k] * tile_b[k][tx];
    }

    workgroupBarrier();
  }

  if (row < uniforms.M && col < uniforms.N) {
    let idx_c = uniforms.offset_c + row * uniforms.stride_c_m + col * uniforms.stride_c_n;
    var result = uniforms.alpha * acc;
    if (uniforms.beta != 0.0) {
      result += uniforms.beta * matrix_c[idx_c];
    }
    ${hasBias ? 'result += bias[col];' : ''}
    matrix_c[idx_c] = result;
  }
}
''';

    return WgslShaderModule(
      name: 'tiled_matmul_${tileSize}x$tileSize${hasBias ? "_bias" : ""}',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {
        'tileSize': tileSize,
        'hasBias': hasBias,
        'dtype': dtype.wgslType,
      },
    );
  }

  /// Generates a batched GEMM compute shader for 3D/ND batched tensors (B x M x N).
  static WgslShaderModule batchedMatmul({
    int tileSize = 16,
    WgslDType dtype = WgslDType.float32,
  }) {
    final typeName = dtype.wgslType;
    final wgSize = WgslWorkgroupSize(tileSize, tileSize, 1);

    final bindings = [
      WgslBinding(
        group: 0,
        binding: 0,
        name: 'matrix_a',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 1,
        name: 'matrix_b',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 2,
        name: 'matrix_c',
        dtype: dtype,
        access: WgslBufferAccess.readWrite,
      ),
      WgslBinding(
        group: 0,
        binding: 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'BatchedMatmulUniforms',
      ),
    ];

    final code =
        '''
// WGSL Batched GEMM (B x M x N) = (B x M x K) x (B x K x N)
struct BatchedMatmulUniforms {
  M: u32,
  N: u32,
  K: u32,
  batch_size: u32,
  batch_stride_a: u32,
  batch_stride_b: u32,
  batch_stride_c: u32,
  alpha: f32,
}

${bindings.map((b) => b.toWgslDeclaration()).join('\n')}

var<workgroup> tile_a: array<array<$typeName, $tileSize>, $tileSize>;
var<workgroup> tile_b: array<array<$typeName, $tileSize>, $tileSize>;

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let col = global_id.x;
  let row = global_id.y;
  let batch_idx = global_id.z;
  let tx = local_id.x;
  let ty = local_id.y;

  if (batch_idx >= uniforms.batch_size) {
    return;
  }

  let batch_off_a = batch_idx * uniforms.batch_stride_a;
  let batch_off_b = batch_idx * uniforms.batch_stride_b;
  let batch_off_c = batch_idx * uniforms.batch_stride_c;

  var acc: $typeName = 0.0;
  let num_tiles = (uniforms.K + ${tileSize}u - 1u) / ${tileSize}u;

  for (var t = 0u; t < num_tiles; t++) {
    let tiled_k_a = t * ${tileSize}u + tx;
    if (row < uniforms.M && tiled_k_a < uniforms.K) {
      tile_a[ty][tx] = matrix_a[batch_off_a + row * uniforms.K + tiled_k_a];
    } else {
      tile_a[ty][tx] = 0.0;
    }

    let tiled_k_b = t * ${tileSize}u + ty;
    if (tiled_k_b < uniforms.K && col < uniforms.N) {
      tile_b[ty][tx] = matrix_b[batch_off_b + tiled_k_b * uniforms.N + col];
    } else {
      tile_b[ty][tx] = 0.0;
    }

    workgroupBarrier();

    for (var k = 0u; k < ${tileSize}u; k++) {
      acc += tile_a[ty][k] * tile_b[k][tx];
    }

    workgroupBarrier();
  }

  if (row < uniforms.M && col < uniforms.N) {
    matrix_c[batch_off_c + row * uniforms.N + col] = uniforms.alpha * acc;
  }
}
''';

    return WgslShaderModule(
      name: 'batched_matmul_${tileSize}x$tileSize',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {'tileSize': tileSize, 'dtype': dtype.wgslType},
    );
  }

  /// Generates a 2D Convolution compute shader (NCHW format).
  static WgslShaderModule conv2d({
    WgslDType dtype = WgslDType.float32,
    bool hasBias = false,
  }) {
    final typeName = dtype.wgslType;
    final wgSize = WgslWorkgroupSize.tiled2D;

    final bindings = [
      WgslBinding(
        group: 0,
        binding: 0,
        name: 'input',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 1,
        name: 'weight',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 2,
        name: 'output',
        dtype: dtype,
        access: WgslBufferAccess.readWrite,
      ),
      if (hasBias)
        WgslBinding(
          group: 0,
          binding: 3,
          name: 'bias',
          dtype: dtype,
          access: WgslBufferAccess.read,
        ),
      WgslBinding(
        group: 0,
        binding: hasBias ? 4 : 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'Conv2dUniforms',
      ),
    ];

    final code =
        '''
// WGSL 2D Convolution (NCHW)
struct Conv2dUniforms {
  batch_size: u32,
  in_channels: u32,
  in_height: u32,
  in_width: u32,
  out_channels: u32,
  out_height: u32,
  out_width: u32,
  kernel_h: u32,
  kernel_w: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
}

${bindings.map((b) => b.toWgslDeclaration()).join('\n')}

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let out_x = global_id.x; // out_width
  let out_y = global_id.y; // out_height
  let out_cz = global_id.z; // out_channel + n * out_channels

  if (out_x >= uniforms.out_width || out_y >= uniforms.out_height) {
    return;
  }

  let total_cz = uniforms.batch_size * uniforms.out_channels;
  if (out_cz >= total_cz) {
    return;
  }

  let n = out_cz / uniforms.out_channels;
  let oc = out_cz % uniforms.out_channels;

  let group_id = oc / (uniforms.out_channels / uniforms.groups);
  let channels_per_group = uniforms.in_channels / uniforms.groups;
  let in_c_start = group_id * channels_per_group;

  var sum: $typeName = 0.0;

  for (var ic_rel = 0u; ic_rel < channels_per_group; ic_rel++) {
    let ic = in_c_start + ic_rel;
    for (var kh = 0u; kh < uniforms.kernel_h; kh++) {
      let in_y = i32(out_y * uniforms.stride_h + kh * uniforms.dilation_h) - i32(uniforms.pad_h);
      if (in_y < 0 || in_y >= i32(uniforms.in_height)) {
        continue;
      }
      for (var kw = 0u; kw < uniforms.kernel_w; kw++) {
        let in_x = i32(out_x * uniforms.stride_w + kw * uniforms.dilation_w) - i32(uniforms.pad_w);
        if (in_x < 0 || in_x >= i32(uniforms.in_width)) {
          continue;
        }

        let in_idx = n * (uniforms.in_channels * uniforms.in_height * uniforms.in_width)
                   + ic * (uniforms.in_height * uniforms.in_width)
                   + u32(in_y) * uniforms.in_width
                   + u32(in_x);

        let weight_idx = oc * (channels_per_group * uniforms.kernel_h * uniforms.kernel_w)
                       + ic_rel * (uniforms.kernel_h * uniforms.kernel_w)
                       + kh * uniforms.kernel_w
                       + kw;

        sum += input[in_idx] * weight[weight_idx];
      }
    }
  }

  ${hasBias ? 'sum += bias[oc];' : ''}

  let out_idx = n * (uniforms.out_channels * uniforms.out_height * uniforms.out_width)
              + oc * (uniforms.out_height * uniforms.out_width)
              + out_y * uniforms.out_width
              + out_x;

  output[out_idx] = sum;
}
''';

    return WgslShaderModule(
      name: 'conv2d${hasBias ? "_bias" : ""}',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {'hasBias': hasBias, 'dtype': dtype.wgslType},
    );
  }

  /// Generates a bank-conflict-free 2D transpose compute shader using shared memory.
  static WgslShaderModule transpose({
    int tileSize = 16,
    WgslDType dtype = WgslDType.float32,
  }) {
    final typeName = dtype.wgslType;
    final wgSize = WgslWorkgroupSize(tileSize, tileSize, 1);

    final bindings = [
      WgslBinding(
        group: 0,
        binding: 0,
        name: 'src',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 1,
        name: 'dst',
        dtype: dtype,
        access: WgslBufferAccess.readWrite,
      ),
      WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'TransposeUniforms',
      ),
    ];

    final code =
        '''
// WGSL 2D Coalesced Transpose Matrix
struct TransposeUniforms {
  rows: u32,
  cols: u32,
  stride_src_r: u32,
  stride_src_c: u32,
  stride_dst_r: u32,
  stride_dst_c: u32,
  offset_src: u32,
  offset_dst: u32,
}

${bindings.map((b) => b.toWgslDeclaration()).join('\n')}

// Padding +1 prevents bank conflicts in shared memory
var<workgroup> tile: array<array<$typeName, ${tileSize + 1}>, $tileSize>;

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let in_col = workgroup_id.x * ${tileSize}u + local_id.x;
  let in_row = workgroup_id.y * ${tileSize}u + local_id.y;

  if (in_row < uniforms.rows && in_col < uniforms.cols) {
    let src_idx = uniforms.offset_src + in_row * uniforms.stride_src_r + in_col * uniforms.stride_src_c;
    tile[local_id.y][local_id.x] = src[src_idx];
  }

  workgroupBarrier();

  let out_col = workgroup_id.y * ${tileSize}u + local_id.x;
  let out_row = workgroup_id.x * ${tileSize}u + local_id.y;

  if (out_row < uniforms.cols && out_col < uniforms.rows) {
    let dst_idx = uniforms.offset_dst + out_row * uniforms.stride_dst_r + out_col * uniforms.stride_dst_c;
    dst[dst_idx] = tile[local_id.x][local_id.y];
  }
}
''';

    return WgslShaderModule(
      name: 'transpose_${tileSize}x$tileSize',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {'tileSize': tileSize, 'dtype': dtype.wgslType},
    );
  }

  /// Generates a numerically stable Softmax WGSL compute shader along the last dimension.
  static WgslShaderModule softmax({
    int workgroupSize = 256,
    WgslDType dtype = WgslDType.float32,
  }) {
    final typeName = dtype.wgslType;
    final wgSize = WgslWorkgroupSize(workgroupSize, 1, 1);

    final bindings = [
      WgslBinding(
        group: 0,
        binding: 0,
        name: 'src',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 1,
        name: 'dst',
        dtype: dtype,
        access: WgslBufferAccess.readWrite,
      ),
      WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'SoftmaxUniforms',
      ),
    ];

    final code =
        '''
// WGSL Numerically Stable Softmax (Last Axis)
struct SoftmaxUniforms {
  num_rows: u32,
  row_size: u32,
  pad0: u32,
  pad1: u32,
}

${bindings.map((b) => b.toWgslDeclaration()).join('\n')}

var<workgroup> s_max: array<$typeName, $workgroupSize>;
var<workgroup> s_sum: array<$typeName, $workgroupSize>;

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = workgroup_id.x;
  let tid = local_id.x;
  if (row >= uniforms.num_rows) {
    return;
  }

  let row_offset = row * uniforms.row_size;

  // 1. Compute maximum in row for numerical stability
  var local_max: $typeName = -3.402823e+38;
  var i = tid;
  while (i < uniforms.row_size) {
    local_max = max(local_max, src[row_offset + i]);
    i += ${workgroupSize}u;
  }
  s_max[tid] = local_max;
  workgroupBarrier();

  for (var s = ${workgroupSize ~/ 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      s_max[tid] = max(s_max[tid], s_max[tid + s]);
    }
    workgroupBarrier();
  }
  let row_max = s_max[0];
  workgroupBarrier();

  // 2. Compute sum of exponentials
  var local_sum: $typeName = 0.0;
  i = tid;
  while (i < uniforms.row_size) {
    local_sum += exp(src[row_offset + i] - row_max);
    i += ${workgroupSize}u;
  }
  s_sum[tid] = local_sum;
  workgroupBarrier();

  for (var s = ${workgroupSize ~/ 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      s_sum[tid] = s_sum[tid] + s_sum[tid + s];
    }
    workgroupBarrier();
  }
  let row_sum = s_sum[0];
  workgroupBarrier();

  // 3. Normalize values
  i = tid;
  while (i < uniforms.row_size) {
    dst[row_offset + i] = exp(src[row_offset + i] - row_max) / row_sum;
    i += ${workgroupSize}u;
  }
}
''';

    return WgslShaderModule(
      name: 'softmax_last_axis',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {'workgroupSize': workgroupSize, 'dtype': dtype.wgslType},
    );
  }

  /// Generates a Root Mean Square Layer Normalization (RMSNorm) WGSL compute shader.
  static WgslShaderModule rmsNorm({
    int workgroupSize = 256,
    WgslDType dtype = WgslDType.float32,
  }) {
    final typeName = dtype.wgslType;
    final wgSize = WgslWorkgroupSize(workgroupSize, 1, 1);

    final bindings = [
      WgslBinding(
        group: 0,
        binding: 0,
        name: 'src',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 1,
        name: 'weight',
        dtype: dtype,
        access: WgslBufferAccess.read,
      ),
      WgslBinding(
        group: 0,
        binding: 2,
        name: 'dst',
        dtype: dtype,
        access: WgslBufferAccess.readWrite,
      ),
      WgslBinding(
        group: 0,
        binding: 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'RMSNormUniforms',
      ),
    ];

    final code =
        '''
// WGSL RMSNorm (Root Mean Square Layer Normalization)
struct RMSNormUniforms {
  num_rows: u32,
  dim: u32,
  eps: f32,
  pad0: u32,
}

${bindings.map((b) => b.toWgslDeclaration()).join('\n')}

var<workgroup> s_sq: array<$typeName, $workgroupSize>;

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = workgroup_id.x;
  let tid = local_id.x;
  if (row >= uniforms.num_rows) {
    return;
  }

  let row_offset = row * uniforms.dim;

  // 1. Compute sum of squares
  var local_sq: $typeName = 0.0;
  var i = tid;
  while (i < uniforms.dim) {
    let v = src[row_offset + i];
    local_sq += v * v;
    i += ${workgroupSize}u;
  }
  s_sq[tid] = local_sq;
  workgroupBarrier();

  for (var s = ${workgroupSize ~/ 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      s_sq[tid] = s_sq[tid] + s_sq[tid + s];
    }
    workgroupBarrier();
  }
  let mean_sq = s_sq[0] / f32(uniforms.dim);
  let inv_rms = inverseSqrt(mean_sq + uniforms.eps);
  workgroupBarrier();

  // 2. Normalize and apply scale weights
  i = tid;
  while (i < uniforms.dim) {
    dst[row_offset + i] = src[row_offset + i] * inv_rms * weight[i];
    i += ${workgroupSize}u;
  }
}
''';

    return WgslShaderModule(
      name: 'rmsnorm_last_axis',
      code: code,
      workgroupSize: wgSize,
      bindings: bindings,
      metadata: {'workgroupSize': workgroupSize, 'dtype': dtype.wgslType},
    );
  }
}
